import torch

from .norm import normalize

FUNCTION_NAME = "Styblinski-Tang"
START_POS = torch.tensor([4.45, 4.1])
EVAL_SIZE = ((-5, 5), (-5, 5))
GLOBAL_MINIMUM_LOC = torch.tensor([[-2.903534, -2.903534]])


@normalize(-156.65626525878906, 917.125)
@torch.jit.script
def stybtang(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Styblinski–Tang function.

    Args:
        x (torch.Tensor): A tensor with last dimension size 2, representing [x, y].

    Returns:
        torch.Tensor: Scalar tensor with the Styblinski–Tang function value.
    """
    y = torch.sum(x**4 - 16 * x**2 + 5 * x)
    return y
