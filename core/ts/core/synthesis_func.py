# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import math
from typing import List, OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SynthesisConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        residual: bool = False,
    ):
        
        super().__init__()
        self.pad = int((kernel_size - 1) / 2)
        self.residual = residual
        self.in_channels,self.out_channels = in_channels, out_channels
        self.kernel_size = kernel_size

       

    def forward(self, x: Tensor, weight:Tensor, bias:Tensor) -> Tensor:
        padded_x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="replicate")
        y = F.conv2d(padded_x, weight, bias)

        if self.residual:
            return y + x
        else:
            return y

    def initialize_parameters(self) -> None:
        bias = nn.Parameter(torch.zeros(self.out_channels,dtype=torch.float32), requires_grad=True)

        if self.residual:
            weight = nn.Parameter(
                torch.zeros((self.out_channels,self.in_channels,self.kernel_size,self.kernel_size),dtype=torch.float32), requires_grad=True
            )
        else:
            k = 1 / (self.in_channels * self.kernel_size * self.kernel_size)
            sqrt_k = math.sqrt(k)

            weight = nn.Parameter(
                (torch.rand((self.out_channels,self.in_channels,self.kernel_size,self.kernel_size),dtype=torch.float32) - 0.5) * 2 * sqrt_k / (self.out_channels**2),
                requires_grad=True,
            )
        return weight,bias

class Synthesis(nn.Module):
    
    possible_non_linearity = {
        "none": nn.Identity,
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU
    }

    possible_mode = ["linear", "residual"]

    def __init__(self, input_ft: int, layers_dim: List[str]):
        super().__init__()
        layers_list = nn.ModuleList()

        # Construct the hidden layer(s)
        for layers in layers_dim:
            out_ft, k_size, mode, non_linearity = layers.split("-")
            out_ft = int(out_ft)
            k_size = int(k_size)

            # Check that mode and non linearity is correct
            assert (
                mode in Synthesis.possible_mode
            ), f"Unknown mode. Found {mode}. Should be in {Synthesis.possible_mode}"

            assert non_linearity in Synthesis.possible_non_linearity, (
                f"Unknown non linearity. Found {non_linearity}. "
                f"Should be in {Synthesis.possible_non_linearity.keys()}"
            )

            layers_list.append(
                SynthesisConv2d(input_ft, out_ft, k_size, residual=mode == "residual")
            )
            layers_list.append(Synthesis.possible_non_linearity[non_linearity]())

            input_ft = out_ft

        self.mlp = layers_list#nn.Sequential(*layers_list)
        self.param_dict = None

    def forward(self, x: Tensor,weights:nn.ParameterList) -> Tensor:
        for layer_idx, layer in enumerate(self.mlp):
            if isinstance(layer, SynthesisConv2d):
                pid = self.param_dict[layer_idx]
                #print(pid)
                x = layer(x,weights[pid],weights[pid+1])
            else:
                x = layer(x)
        return x

    def initialize_parameters(self) -> nn.ParameterList:
        param_list = []
        self.param_dict = {}
        idx = 0
        for layer_idx, layer in enumerate(self.mlp):
            if isinstance(layer, SynthesisConv2d):
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
            if isinstance(layer, SynthesisConv2d):
                self.param_dict[layer_idx] = idx
                idx += 2
    
if __name__ == '__main__':
    ts = Synthesis(10,["40-1-linear-relu","3-1-linear-none","3-3-residual-relu","3-3-residual-none"]).to('cuda:0')
    ts.train()
    dt = torch.rand((10,10,32,32),dtype=torch.float32).to('cuda:0')
    weigts = ts.initialize_parameters()
    weigts = weigts.to('cuda:0')
    a = ts(dt,weigts)
    loss = torch.mean(a**2)
    loss.backward()
    pass
    print(weigts[1].grad)
