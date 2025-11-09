import torch

from .norm import normalize

FUNCTION_NAME = "Schaffer 4"
START_POS = torch.tensor([8.2, 8.4])
EVAL_SIZE = ((-10, 10), (-10, 10))

FUNC_SCALE = 5

GLOBAL_MINIMUM_LOC = torch.tensor(
    [[0.0, 0.125], [0.0, -0.125], [0.125, 0.0], [-0.125, 0.0]]
)


@normalize(-0.2065865695476532, 0.5)
@torch.jit.script
def schaffer4(
    x: torch.Tensor,
    scale: float = FUNC_SCALE,
) -> torch.Tensor:
    """
    Computes the Schaffer function N. 4.

    Args:
        x (torch.Tensor): A tensor with last dimension size 2, representing [x, y].
        scale (torch.Tensor): Scale factor for [x, y].

    Returns:
        torch.Tensor: Scalar tensor with the Schaffer function value.
    """
    x = x * scale
    x1, x2 = x[0], x[1]

    fact1 = (torch.cos(torch.sin(torch.abs(x1**2 - x2**2)))) ** 2 - 0.5
    fact2 = (1 + 0.001 * (x1**2 + x2**2)) ** 2
    return fact1 / fact2
