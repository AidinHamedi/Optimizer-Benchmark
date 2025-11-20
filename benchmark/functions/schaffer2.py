import torch

from .norm import normalize

FUNCTION_NAME = "Schaffer 2"
START_POS = torch.tensor([8.2, 8.4])
EVAL_SIZE = ((-10, 10), (-10, 10))
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])

FUNC_SCALE = 5


@normalize(-0.5, 0.4964832067489624)
@torch.jit.script
def schaffer2(
    x: torch.Tensor,
    scale: float = FUNC_SCALE,
) -> torch.Tensor:
    """
    Computes the Schaffer function N. 2.

    Args:
        x (torch.Tensor): A tensor with last dimension size 2, representing [x, y].
        scale (torch.Tensor): Scale factor for [x, y].

    Returns:
        torch.Tensor: Scalar tensor with the Schaffer function value.
    """
    x = x * scale
    x1, x2 = x[0], x[1]

    fact1 = torch.sin(x1**2 - x2**2) ** 2 - 0.5
    fact2 = (1 + 0.001 * (x1**2 + x2**2)) ** 2

    return fact1 / fact2
