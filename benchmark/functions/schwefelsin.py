import torch

from .norm import normalize

FUNCTION_NAME = "SchwefelSin"
START_POS = torch.tensor([11.4, 12])
EVAL_SIZE = ((-23, 23), (-23, 23))
GLOBAL_MINIMUM_LOC = torch.tensor(
    [
        [-22.37689208984375, -22.37689208984375],
        [-22.37689208984375, 16.8],
        [16.8, -22.37689208984375],
    ]
)

FUNC_SCALE = 500.0 / 20.0


@normalize(-1114.2998046875, 1114.2998046875)
@torch.jit.script
def schwefel_sin(
    x: torch.Tensor,
    scale: float = FUNC_SCALE,
) -> torch.Tensor:
    """Compute the Schwefel function.

    Args:
        x: Input tensor of shape [2] representing [x, y] coordinates.
        scale: Scaling factor applied to input (default: 25.0).

    Returns:
        Scalar tensor with the function value.
    """
    x = x * scale

    return -torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))))
