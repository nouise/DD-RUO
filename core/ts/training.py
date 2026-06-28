# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import math
from dataclasses import dataclass, field
from typing import Optional, Union,Any, Dict
from dataclasses import dataclass, field, fields
import torch
from torch import Tensor
import copy

@dataclass(kw_only=True)
class LossFunctionOutput():
    loss: Optional[float] = None                                        # The RD cost to optimize
    mse: Optional[float] = None                                         # Mean squared error                     [ / ]
    rate_nn_bpp: Optional[float] = None                                 # Rate associated to the neural networks [bpp]
    rate_latent_bpp: Optional[float] = None                             # Rate associated to the latent          [bpp]

    psnr_db: Optional[float] = field(init=False, default=None)          # PSNR                                  [ dB]
    total_rate_bpp: Optional[float] = field(init=False, default=None)   # Overall rate: latent & NNs            [bpp]

    def __post_init__(self):
        if self.mse is not None:
            self.psnr_db = -10.0 * math.log10(self.mse)

        if self.rate_nn_bpp is not None and self.rate_latent_bpp is not None:
            self.total_rate_bpp = self.rate_nn_bpp + self.rate_latent_bpp
            
    def pretty_string(
        self,
        show_col_name:bool = True,
        additional_data: Dict[str, Any] = {},
    ) -> str:
        col_name = ""
        values = ""
        COL_WIDTH = 10
        INTER_COLUMN_SPACE = " "

        for k in fields(self):

            val = copy.deepcopy(getattr(self, k.name))
            if val is None:
                continue
            col_name += f"{self._format_column_name(k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}"
            values += f"{self._format_value(val, attribute_name=k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}"

        for k, v in additional_data.items():
            col_name += f"{k:<{COL_WIDTH}}{INTER_COLUMN_SPACE}"
            values += f"{v:<{COL_WIDTH}}{INTER_COLUMN_SPACE}"
            
        if show_col_name:
            return col_name + "\n" + values
        else:
            return values
        
    def _format_column_name(self, col_name: str) -> str:
        # Syntax: {'long_name': 'short_name'}
        LONG_TO_SHORT = {
            "rate_latent_bpp": "latent_bpp",
            "rate_nn_bpp": "nn_bpp",
            "encoding_time_second": "time_sec",
            "encoding_iterations_cnt": "itr",
            "alpha_mean": "alpha",
            "beta_mean": "beta",
            "prediction_psnr_db": "pred_db",
            "dummy_prediction_psnr_db": "dummy_pred",
        }
        
        if col_name not in LONG_TO_SHORT:
            return col_name
        else:
            return LONG_TO_SHORT.get(col_name)
        
    def _format_value(
        self, value: Union[str, int, float, Tensor], attribute_name: str = ""
    ) -> str:
        if attribute_name == "loss":
            value *= 1000

        if attribute_name == "img_size":
            value = "x".join([str(tmp) for tmp in value])

        if isinstance(value, str):
            return value
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            return f"{value:.6f}"
        elif isinstance(value, Tensor):
            return f"{value.item():.6f}"


def loss_function(decoded_image: Tensor, rate_latent_bit: Tensor, target_image: Tensor, lmbda: float = 1e-3) -> LossFunctionOutput:
    

    mse = torch.mean((decoded_image-target_image)**2)

    n_pixels = decoded_image.size()[0] * decoded_image.size()[-2] * decoded_image.size()[-1]
   
    rate_bpp = rate_latent_bit.sum() / n_pixels

    loss = mse + lmbda * rate_bpp

    output = LossFunctionOutput(loss=loss, mse=mse.detach().item(), rate_latent_bpp=rate_latent_bit.detach().sum().item() / n_pixels, )

    return output
