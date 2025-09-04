import torch

from .norm import normalize

START_POS = torch.tensor([4.2, 5.1])
EVAL_SIZE = ((-5.9, 5.9), (-5.9, 5.9))
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])


@normalize(-0.9469, -2.6508e-08)
@torch.jit.script
def drop_wave(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the Drop-Wave function.

    Args:
        x (torch.Tensor): A tensor with last dimension size 2, representing [x, y].

    Returns:
        torch.Tensor: Scalar tensor with the Drop-Wave function value.
    """
    r2 = torch.sum(x**2, dim=-1)
    frac1 = 1 + torch.cos(12 * torch.sqrt(r2))
    frac2 = 0.5 * r2 + 2
    return -frac1 / frac2
