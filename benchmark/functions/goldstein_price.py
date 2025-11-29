import torch

from .norm import normalize

FUNCTION_NAME = "Goldstein-Price"
START_POS = torch.tensor([-1.8, 1.8])
EVAL_SIZE = ((-2, 2), (-2, 2))
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, -1.0]])


@normalize(3.0231451988220215, 1457606.625)
@torch.jit.script
def goldstein_price(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Goldstein-Price function.

    Args:
        x (torch.Tensor): A tensor with last dimension size 2, representing [x, y].

    Returns:
        torch.Tensor: Scalar tensor with the function value.
    """
    x1 = x[..., 0]
    x2 = x[..., 1]

    fact1a = (x1 + x2 + 1) ** 2
    fact1b = 19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
    fact1 = 1 + fact1a * fact1b

    fact2a = (2 * x1 - 3 * x2) ** 2
    fact2b = 18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
    fact2 = 30 + fact2a * fact2b

    return fact1 * fact2
