import torch

from .norm import normalize

FUNCTION_NAME = "Schaffer 2"
START_POS = torch.tensor([8.2, 8.4])
EVAL_SIZE = ((-10, 10), (-10, 10))
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])


@normalize(8.0585e-05, 0.9961)
@torch.jit.script
def schaffer2(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Schaffer function N. 2.

    Args:
        x (torch.Tensor): A 1D tensor with two elements representing [x, y].

    Returns:
        torch.Tensor: Scalar tensor with the Schaffer function value.
    """
    # To make the eval size smaller
    x1 = x[0] * 5
    x2 = x[1] * 5

    fact1 = torch.sin(x1**2 - x2**2) ** 2 - 0.5
    fact2 = (1 + 0.001 * (x1**2 + x2**2)) ** 2

    return 0.5 + fact1 / fact2
