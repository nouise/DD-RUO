from typing import Optional
import torch
from torch import Tensor, nn

MAX_ARM_MASK_SIZE = 9

POSSIBLE_Q_STEP_SHIFT = {
    "arm": {
        "weight": torch.linspace(-8, 0, 9, device="cpu"),
        "bias": torch.linspace(-16, 0, 17, device="cpu"),
    },
}
POSSIBLE_Q_STEP = {
    "arm": {
        "weight": 2.0 ** POSSIBLE_Q_STEP_SHIFT["arm"]["weight"],
        "bias": 2.0 ** POSSIBLE_Q_STEP_SHIFT["arm"]["bias"],
    },
    "upsampling": {
        "weight": 2.0 ** torch.linspace(-15, 0, 16, device="cpu"),
        "bias": 2.0 ** torch.tensor([0.]),
    },
    "synthesis": {
        "weight": 2.0 ** torch.linspace(-15, 0, 16, device="cpu"),
        "bias": 2.0 ** torch.linspace(-15, 0, 16, device="cpu"),
    },
}

POSSIBLE_EXP_GOL_COUNT = {
    "arm": {
        "weight": torch.linspace(0, 12, 13, device="cpu"),
        "bias": torch.linspace(0, 12, 13, device="cpu"),
    },
    "upsampling": {
        "weight": torch.linspace(0, 12, 13, device="cpu"),
        "bias": torch.linspace(0, 12, 13, device="cpu"),
    },
    "synthesis": {
        "weight": torch.linspace(0, 12, 13, device="cpu"),
        "bias": torch.linspace(0, 12, 13, device="cpu"),
    },
}

FIXED_POINT_FRACTIONAL_BITS = 8  # 8 works fine in pure int mode
FIXED_POINT_FRACTIONAL_MULT = 2**FIXED_POINT_FRACTIONAL_BITS

MAX_AC_MAX_VAL = 65535  # 2**16 for 16-bit code in bitstream header.


def get_q_step_from_parameter_name(
    parameter_name: str, q_step
) -> Optional[float]:
    """Return the specific quantization step from q_step (a dictionary
    with several quantization steps). The specific quantization step is
    selected through the parameter name.

    Args:
        parameter_name (str): Name of the parameter in the state dict.
        q_step (DescriptorNN): Dictionary gatherting several quantization
            steps. E.g. one quantization step for the weights and one for
            the biases.

    Returns:
        Optional[float]: The quantization step associated to the parameter.
            Return None if nothing is found.
    """
    if parameter_name.endswith(".weight"):
        current_q_step = q_step.get("weight")
    elif parameter_name.endswith(".bias"):
        current_q_step = q_step.get("bias")
    else:
        print(
            'Parameter name should end with ".weight" or ".bias" '
            f"Found: {parameter_name}"
        )
        current_q_step = None

    return current_q_step


@torch.no_grad()
def measure_expgolomb_rate(q_module: nn.Module, q_step, expgol_cnt):
    """Get the rate associated with the current parameters.

    Returns:
        DescriptorNN: The rate of the different modules wrapped inside a dictionary
            of float. It does **not** return tensor so no back propagation is possible
    """
    # Concatenate the sent parameters here to measure the entropy later
    sent_param = {"bias": [], "weight": []}
    rate_param = {"bias": 0.0, "weight": 0.0}

    param = q_module.get_param()
    # Retrieve all the sent item
    for parameter_name, parameter_value in param.items():
        current_q_step = get_q_step_from_parameter_name(parameter_name, q_step)
        # Current quantization step is None because the module is not yet
        # quantized. Return an all zero rate
        if current_q_step is None:
            return rate_param

        # Quantization is round(parameter_value / q_step) * q_step so we divide by q_step
        # to obtain the sent latent.
        current_sent_param = (parameter_value / current_q_step).view(-1)

        if parameter_name.endswith(".weight"):
            sent_param["weight"].append(current_sent_param)
        elif parameter_name.endswith(".bias"):
            sent_param["bias"].append(current_sent_param)
        else:
            print(
                'Parameter name should end with ".weight" or ".bias" '
                f"Found: {parameter_name}"
            )
            return rate_param

    # For each sent parameters (e.g. all biases and all weights)
    # compute their cost with an exp-golomb coding.
    for k, v in sent_param.items():
        # If we do not have any parameter, there is no rate associated.
        # This can happens for the upsampling biases for instance
        if len(v) == 0:
            rate_param[k] = 0.0
            continue

        # Current exp-golomb count is None because the module is not yet
        # quantized. Return an all zero rate
        current_expgol_cnt = expgol_cnt[k]
        if current_expgol_cnt is None:
            return rate_param

        # Concatenate the list of parameters as a big one dimensional tensor
        v = torch.cat(v)

        # This will be pretty long! Could it be vectorized?
        rate_param[k] = exp_golomb_nbins(v, count = current_expgol_cnt)

    return rate_param


def exp_golomb_nbins(symbol: Tensor, count: int = 0) -> Tensor:
    """Compute the number of bits required to encode a Tensor of integers
    using an exponential-golomb code with exponent ``count``.

    Args:
        symbol: Tensor to encode
        count (int, optional): Exponent of the exp-golomb code. Defaults to 0.

    Returns:
        Number of bits required to encode all the symbols.
    """

    # We encode the sign equiprobably at the end thus one more bit if symbol != 0
    nbins = 2 * torch.floor(torch.log2(symbol.abs() / (2 ** count) + 1)) + count + 1 + (symbol != 0)
    res = nbins.sum()
    return res

