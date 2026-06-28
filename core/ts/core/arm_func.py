# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


from typing import OrderedDict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, index_select, nn


class ArmLinear(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool = False,
    ):

        super().__init__()
        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels

    def initialize_parameters(self):
        bias = nn.Parameter(torch.zeros(self.out_channels,dtype=torch.float32), requires_grad=True)
        if self.residual:
            weight = torch.zeros((self.out_channels, self.in_channels),dtype=torch.float32)
        else:
            out_channel = self.out_channels
            weight = torch.randn((self.out_channels, self.in_channels),dtype=torch.float32) / out_channel**2
        weight = nn.Parameter(weight, requires_grad=True)
        return weight,bias
        
    def forward(self, x: Tensor, weight:Tensor, bias:Tensor) -> Tensor:
        if self.residual:
            return F.linear(x, weight, bias) + x
        else:
            return F.linear(x, weight, bias=bias)

class ArmIntLinear(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fpfm: int = 0,
        pure_int: bool = False,
        residual: bool = False,
    ):

        super().__init__()

        self.fpfm = fpfm
        self.pure_int = pure_int
        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:

        if self.residual:
            xx = F.linear(x, self.weight, bias=self.bias) + x*self.fpfm
        else:
            xx = F.linear(x, self.weight, bias=self.bias)

        # Renorm by fpfm after our (x*fpfm)*(qw*fpfm) multiplication.
        # WE MAKE INTEGER DIVISION OBEY C++ (TO-ZERO) SEMANTICS, NOT PYTHON (TO-NEGATIVE-INFINITY) SEMANTICS
        if self.pure_int:
            xx = xx + torch.sign(xx)*self.fpfm//2
            # We separate out -ve and non-ve.
            neg_result = -((-xx)//self.fpfm)
            pos_result = xx//self.fpfm
            result = torch.where(xx < 0, neg_result, pos_result)
        else:
            xx = xx + torch.sign(xx)*self.fpfm/2
            # We separate out -ve and non-ve.
            neg_result = -((-xx)/self.fpfm)
            pos_result = xx/self.fpfm
            result = torch.where(xx < 0, neg_result, pos_result)
            result = result.to(torch.int32).to(torch.float)

        return result

class Arm(nn.Module):
    
    def __init__(self, dim_arm: int, n_hidden_layers_arm: int):
        super().__init__()

        assert dim_arm % 8 == 0, (
            f"ARM context size and hidden layer dimension must be "
            f"a multiple of 8. Found {dim_arm}."
        )

        # ======================== Construct the MLP ======================== #
        layers_list = nn.ModuleList()

        # Construct the hidden layer(s)
        for i in range(n_hidden_layers_arm):
            layers_list.append(ArmLinear(dim_arm, dim_arm, residual=True))
            layers_list.append(nn.ReLU())

        # Construct the output layer. It always has 2 outputs (mu and scale)
        layers_list.append(ArmLinear(dim_arm, 2, residual=False))
        self.mlp = layers_list
        # ======================== Construct the MLP ======================== #

    def forward(self, x: Tensor, weights) -> Tuple[Tensor, Tensor, Tensor]:
        raw_proba_param = x
        for layer_idx, layer in enumerate(self.mlp):
            if isinstance(layer, ArmLinear):
                pid = self.param_dict[layer_idx]
                raw_proba_param = layer(raw_proba_param,weights[pid],weights[pid+1])
            else:
                raw_proba_param = layer(raw_proba_param)
        
        mu = raw_proba_param[:, 0]
        log_scale = raw_proba_param[:, 1]

        scale = torch.exp(torch.clamp(log_scale - 4, min=-4.6, max=5.0))

        return mu, scale, log_scale

    def initialize_parameters(self) -> nn.ParameterList:
        param_list = []
        self.param_dict = {}
        idx = 0
        for layer_idx, layer in enumerate(self.mlp):
            if isinstance(layer, ArmLinear):
                wt,bias = layer.initialize_parameters()
                param_list.append(wt)
                param_list.append(bias)
                self.param_dict[layer_idx] = idx
                idx += 2
        return nn.ParameterList(param_list)
    
    def initialize_parameters_map(self) -> None:
        self.param_dict = {}
        idx = 0
        for layer_idx, layer in enumerate(self.mlp):
            if isinstance(layer, ArmLinear):
                self.param_dict[layer_idx] = idx
                idx += 2

class ArmInt(nn.Module):

    def __init__(self, dim_arm: int, n_hidden_layers_arm: int, fpfm: int, pure_int: bool):
        super().__init__()

        assert dim_arm % 8 == 0, (
            f"ARM context size and hidden layer dimension must be "
            f"a multiple of 8. Found {dim_arm}."
        )

        self.FPFM = fpfm # fixed-point: multiplication to get int.
        self.pure_int = pure_int # weights and biases are actual int (cpu only), or just int values in floats (gpu friendly).

        # ======================== Construct the MLP ======================== #
        layers_list = nn.ModuleList()

        # Construct the hidden layer(s)
        for i in range(n_hidden_layers_arm):
            layers_list.append(ArmIntLinear(dim_arm, dim_arm, self.FPFM, self.pure_int, residual=True))
            layers_list.append(nn.ReLU())

        # Construct the output layer. It always has 2 outputs (mu and scale)
        layers_list.append(ArmIntLinear(dim_arm, 2, self.FPFM, self.pure_int, residual=False))
        self.mlp = layers_list#nn.Sequential(*layers_list)
        # ======================== Construct the MLP ======================== #

    def transform_param_from_float(self, float_param, param_dict) -> None:
        integerised_param = []
        self.param_dict = param_dict
        for pid, param in enumerate(float_param):
            if pid % 2 == 0:
                float_v = param*self.FPFM
            else:
                float_v = param*self.FPFM*self.FPFM

            float_v = float_v + torch.sign(float_v)*0.5
            neg_result = -(-float_v).to(torch.int32)
            pos_result = float_v.to(torch.int32)
            int_v = torch.where(float_v < 0, neg_result, pos_result)
            if not self.pure_int:
                int_v = int_v.to(torch.float)
            integerised_param.append(nn.parameter.Parameter(int_v, requires_grad=False))
        return integerised_param

    def forward(self, x: Tensor, weights) -> Tuple[Tensor, Tensor, Tensor]:
        xint = x.clone().detach()
        xint = xint*self.FPFM
        if self.pure_int:
            xint = xint.to(torch.int32)

        raw_proba_param = xint
        for layer_idx, layer in enumerate(self.mlp):
            if isinstance(layer, ArmLinear):
                pid = self.param_dict[layer_idx]
                raw_proba_param = layer(raw_proba_param,weights[pid],weights[pid+1])
            else:
                raw_proba_param = layer(raw_proba_param)

        raw_proba_param = raw_proba_param  / self.FPFM

        mu = raw_proba_param[:, 0]
        log_scale = raw_proba_param[:, 1]

        # no scale smaller than exp(-4.6) = 1e-2 or bigger than exp(5.01) = 150
        scale = torch.exp(torch.clamp(log_scale - 4, min=-4.6, max=5.0))

        return mu, scale, log_scale


@torch.jit.script
def _get_neighbor(x: Tensor, mask_size: int, non_zero_pixel_ctx_idx: Tensor) -> Tensor:
    """Use the unfold function to extract the neighbors of each pixel in x.

    Args:
        x (Tensor): [1, 1, H, W] feature map from which we wish to extract the
            neighbors
        mask_size (int): Virtual size of the kernel around the current coded latent.
            mask_size = 2 * n_ctx_rowcol - 1
        non_zero_pixel_ctx_idx (Tensor): [N] 1D tensor containing the indices
            of the non zero context pixels (i.e. floor(N ** 2 / 2) - 1).
            It looks like: [0, 1, ..., floor(N ** 2 / 2) - 1].
            This allows to use the index_select function, which is significantly
            faster than usual indexing.

    Returns:
        torch.tensor: [H * W, floor(N ** 2 / 2) - 1] the spatial neighbors
            the floor(N ** 2 / 2) - 1 neighbors of each H * W pixels.
    """
    pad = int((mask_size - 1) / 2)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)

    # Shape of x_unfold is [B, C, H, W, mask_size, mask_size] --> [B * C * H * W, mask_size * mask_size]
    # reshape is faster than einops.rearrange
    x_unfold = (
        x_pad.unfold(2, mask_size, step=1)
        .unfold(3, mask_size, step=1)
        .reshape(-1, mask_size * mask_size)
    )

    # Convert x_unfold to a 2D tensor: [Number of pixels, all neighbors]
    # This is slower than reshape above
    # x_unfold = rearrange(
    #     x_unfold,
    #     'b c h w mask_h mask_w -> (b c h w) (mask_h mask_w)'
    # )

    # Select the pixels for which the mask is not zero
    # For a N x N mask, select only the first (N x N - 1) / 2 pixels
    # (those which aren't null)
    neighbor = index_select(x_unfold, dim=1, index=non_zero_pixel_ctx_idx)
    return neighbor


@torch.jit.script
def _laplace_cdf(x: Tensor, expectation: Tensor, scale: Tensor) -> Tensor:
    """Compute the laplace cumulative evaluated in x. All parameters
    must have the same dimension.
    Re-implemented here coz it is faster than calling the Laplace distribution
    from torch.distributions.

    Args:
        x (Tensor): Where the cumulative if evaluated.
        expectation (Tensor): Expectation.
        scale (Tensor): Scale

    Returns:
        Tensor: CDF(x, mu, scale)
    """
    shifted_x = x - expectation
    return 0.5 - 0.5 * (shifted_x).sign() * torch.expm1(-(shifted_x).abs() / scale)


def _get_non_zero_pixel_ctx_index(dim_arm: int) -> Tensor:
    """Generate the relative index of the context pixel with respect to the
    actual pixel being decoded.

    1D tensor containing the indices of the non zero context. This corresponds to the one
    in the pattern above. This allows to use the index_select function, which is significantly
    faster than usual indexing.

    0   1   2   3   4   5   6   7   8
    9   10  11  12  13  14  15  16  17
    18  19  20  21  22  23  24  25  26
    27  28  29  30  31  32  33  34  35
    36  37  38  39  *   x   x   x   x
    x   x   x   x   x   x   x   x   x
    x   x   x   x   x   x   x   x   x
    x   x   x   x   x   x   x   x   x
    x   x   x   x   x   x   x   x   x


    Args:
        dim_arm (int): Number of context pixels

    Returns:
        Tensor: 1D tensor with the flattened index of the context pixels.
    """

    if dim_arm == 8:
        return torch.tensor(
            [            13,
                         22,
                     30, 31, 32,
             37, 38, 39, #
            ]
        )

    elif dim_arm == 16:
        return torch.tensor(
            [
                            13, 14,
                    20, 21, 22, 23, 24,
                28, 29, 30, 31, 32, 33,
                37, 38, 39, #
            ]
        )

    elif dim_arm == 24:
        return torch.tensor(
            [
                                4 ,
                        11, 12, 13, 14, 15,
                    19, 20, 21, 22, 23, 24, 25,
                    28, 29, 30, 31, 32, 33, 34,
                36, 37, 38, 39, #
            ]
        )

    elif dim_arm == 32:
        return torch.tensor(
            [
                        2 , 3 , 4 , 5 ,
                    10, 11, 12, 13, 14, 15, 16,
                    19, 20, 21, 22, 23, 24, 25, 26,
                27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, #
            ]
        )

if __name__ == '__main__':
    ts = Arm(24,3).to('cuda:0')
    ts.train()
    dt = torch.empty((100,24),dtype=torch.float32).to('cuda:0')
    weigts = ts.initialize_parameters()
    weigts = weigts.to('cuda:0')
    a,b,c = ts(dt,weigts)
    loss = torch.mean(a**2+b**2+c**2)
    loss.backward()
