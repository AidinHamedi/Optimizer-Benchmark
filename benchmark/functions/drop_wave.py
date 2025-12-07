import torch

from .norm import normalize

FUNCTION_NAME = "Drop-Wave"
START_POS = torch.tensor([4.4, 5.2])
EVAL_SIZE = ((-5.9, 5.9), (-5.9, 5.9))
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])


@normalize(-1.0, -0.0)
@torch.jit.script
def drop_wave(
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute the Drop-Wave function.

    Args:
        x: Input tensor of shape [2] representing [x, y] coordinates.

    Returns:
        Scalar tensor with the function value.
    """
    r2 = torch.sum(x**2, dim=-1)
    frac1 = 1 + torch.cos(12 * torch.sqrt(r2))
    frac2 = 0.5 * r2 + 2
    return -frac1 / frac2
