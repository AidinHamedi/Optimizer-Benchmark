import torch

from .norm import normalize

START_POS = torch.tensor([4.45, 0.13])
EVAL_SIZE = ((-5, 5), (-5, 5))
GLOBAL_MINIMUM_LOC = torch.tensor([[-2.903534, -2.903534]])


@normalize(-78.3310, 750)
@torch.jit.script
def stybtang(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Styblinski–Tang function.

    Args:
        x (torch.Tensor): A tensor representing the input vector.

    Returns:
        torch.Tensor: Scalar tensor with the Styblinski–Tang function value.
    """
    sum_terms = torch.sum(x**4 - 16 * x**2 + 5 * x)
    y = sum_terms / 2.0
    return y
