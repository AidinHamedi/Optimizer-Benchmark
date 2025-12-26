import torch

from .norm import normalize

FUNCTION_NAME = "Styblinski-Tang"
START_POS = torch.tensor([4.65, 4.7])
EVAL_SIZE = ((-5, 5), (-5, 5))
GLOBAL_MINIMUM_LOC = torch.tensor([[-2.903534, -2.903534]])


@normalize(-156.66366577148438, 917.125)
@torch.jit.script
def stybtang(x: torch.Tensor) -> torch.Tensor:
    """Compute the Styblinski-Tang function.

    Args:
        x: Input tensor of shape [2] representing [x, y] coordinates.

    Returns:
        Scalar tensor with the function value.
    """
    y = torch.sum(x**4 - 16 * x**2 + 5 * x)
    return y
