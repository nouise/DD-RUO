# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


from typing import List, OrderedDict

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


class UpsamplingConvTranspose2d(nn.Module):

    kernel_bilinear = torch.tensor(
        [
            [0.0625, 0.1875, 0.1875, 0.0625],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.0625, 0.1875, 0.1875, 0.0625],
        ]
    )

    kernel_bicubic = torch.tensor(
        [
            [ 0.0012359619 , 0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 , 0.0037078857 , 0.0012359619],
            [ 0.0037078857 , 0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 , 0.0111236572 , 0.0037078857],
            [-0.0092010498 ,-0.0276031494 , 0.0684967041 , 0.2300262451 , 0.2300262451 , 0.0684967041 ,-0.0276031494 ,-0.0092010498],
            [-0.0308990479 ,-0.0926971436 , 0.2300262451 , 0.7724761963 , 0.7724761963 , 0.2300262451 ,-0.0926971436 ,-0.0308990479],
            [-0.0308990479 ,-0.0926971436 , 0.2300262451 , 0.7724761963 , 0.7724761963 , 0.2300262451 ,-0.0926971436 ,-0.0308990479],
            [-0.0092010498 ,-0.0276031494 , 0.0684967041 , 0.2300262451 , 0.2300262451 , 0.0684967041 ,-0.0276031494 ,-0.0092010498],
            [ 0.0037078857 , 0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 , 0.0111236572 , 0.0037078857],
            [ 0.0012359619 , 0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 , 0.0037078857 , 0.0012359619],
        ]
    )


    def __init__(
        self,
        upsampling_kernel_size: int,
        static_upsampling_kernel: bool
    ):
        super().__init__()

        assert upsampling_kernel_size >= 4, (
            f"Upsampling kernel size should be >= 4." f"Found {upsampling_kernel_size}"
        )

        assert upsampling_kernel_size % 2 == 0, (
            f"Upsampling kernel size should be even." f"Found {upsampling_kernel_size}"
        )
        self.upsampling_kernel_size = upsampling_kernel_size
        self.initialize_const()

    def initialize_parameters(self) -> Tensor:
        K = self.upsampling_kernel_size
        if K < 8:
            kernel_init = UpsamplingConvTranspose2d.kernel_bilinear
        else:
            kernel_init = UpsamplingConvTranspose2d.kernel_bicubic
        tmpad = (K - kernel_init.size()[0]) // 2
        upsampling_kernel = F.pad(
            kernel_init.clone().detach(),
            (tmpad, tmpad, tmpad, tmpad),
            mode="constant",
            value=0.0,
        )
        upsampling_kernel = rearrange(upsampling_kernel, "k_h k_w -> 1 1 k_h k_w")
        weight = nn.Parameter(upsampling_kernel, requires_grad=True)
        return weight
    
    def initialize_const(self) -> None:
        K = self.upsampling_kernel_size
        self.upsampling_padding = (K // 2, K // 2, K // 2, K // 2)
        self.upsampling_crop = (3 * K - 2) // 2

    def forward(self, x: Tensor, upsampling_weight) -> Tensor:
        x_pad = F.pad(x, self.upsampling_padding, mode="replicate")
        y_conv = F.conv_transpose2d(x_pad, upsampling_weight, stride=2)
        H, W = y_conv.size()[-2:]
        results = y_conv[
            :,
            :,
            self.upsampling_crop : H - self.upsampling_crop,
            self.upsampling_crop : W - self.upsampling_crop,
        ]
        return results


class Upsampling(nn.Module):


    def __init__(self, upsampling_kernel_size: int, static_upsampling_kernel: bool):
        super().__init__()

        self.conv_transpose2d = UpsamplingConvTranspose2d(upsampling_kernel_size, static_upsampling_kernel)

    def forward(self, decoder_side_latent: List[Tensor], weight:nn.ParameterList) -> Tensor:
        latent_reversed = list(reversed(decoder_side_latent))
        upsampled_latent = latent_reversed[0]  # start from smallest
        for target_tensor in latent_reversed[1:]:
            # Our goal is to upsample <upsampled_latent> to the same resolution than <target_tensor>
            x = rearrange(upsampled_latent, "b c h w -> (b c) 1 h w")
            x = self.conv_transpose2d(x,weight[0])
            x = rearrange(x, "(b c) 1 h w -> b c h w", b=upsampled_latent.shape[0])
            # Crop to comply with higher resolution feature maps size before concatenation
            x = x[:, :, : target_tensor.shape[-2], : target_tensor.shape[-1]]
            upsampled_latent = torch.cat((target_tensor, x), dim=1)
        return upsampled_latent


    def initialize_parameters(self) -> nn.ParameterList:
        wt = self.conv_transpose2d.initialize_parameters()
        return nn.ParameterList([wt])

    def initialize_parameters_map(self) -> None:
        pass

if __name__ == '__main__':
    ts = Upsampling(8,False).to('cuda:0')
    ts.train()
    h,w = 2,3
    dt_list = []
    for _ in range(7):
        dt = torch.rand((10,1,h,w),dtype=torch.float32).to('cuda:0')
        h,w = h*2,w*2
        dt_list.append(dt)
    dt_list.reverse()
    weigts = ts.initialize_parameters()
    weigts = weigts.to('cuda:0')
    a = ts(dt_list,weigts)
    loss = torch.mean(a**2)
    loss.backward()
    pass
    print(weigts[0].grad)
