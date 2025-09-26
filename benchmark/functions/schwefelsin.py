import torch

from .norm import normalize

FUNCTION_NAME = "SchwefelSin"
START_POS = torch.tensor([-5, 4])
EVAL_SIZE = ((-23, 23), (-23, 23))
SCALE_TO_ORIGINAL = 500.0 / 20.0
GLOBAL_MINIMUM_LOC = torch.tensor(
    [
        [-22.37689208984375, -22.37689208984375],
        [-22.37689208984375, 16.8],
        [16.8, -22.37689208984375],
    ]
)


@normalize(-1114.2998046875, 1114.2998046875)
@torch.jit.script
def schwefel_sin(
    x: torch.Tensor,
    scale_to_original: torch.Tensor = torch.tensor(SCALE_TO_ORIGINAL),
) -> torch.Tensor:
    """
    Computes the Schwefel function.

    Args:
        x (torch.Tensor): A 1D tensor representing the input vector in 40x40 space.
        scale_to_original (torch.Tensor): Scale factor to map to original ±500 range.

    Returns:
        torch.Tensor: Scalar tensor with the Schwefel function value.
    """
    x_original = x * scale_to_original

    return -torch.sum(x_original * torch.sin(torch.sqrt(torch.abs(x_original))))
