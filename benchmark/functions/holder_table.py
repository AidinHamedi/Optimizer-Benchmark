import torch

from .norm import normalize

START_POS = torch.tensor([-0.001, 0.001])
EVAL_SIZE = ((-10, 10), (-10, 10))
GLOBAL_MINIMUM_LOC = torch.tensor(
    [[8.05502, 9.66459], [8.05502, -9.66459], [-8.05502, 9.66459], [-8.05502, -9.66459]]
)


@normalize(-56.49380874633789, -1.8016504327533767e-05)
@torch.jit.script
def holder_table(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Holder Table function.

    Args:
        x (torch.Tensor): A tensor with last dimension size 2, representing [x1, x2].

    Returns:
        torch.Tensor: Scalar tensor with the Holder Table function value.
    """
    x1 = x[..., 0]
    x2 = x[..., 1]

    fact1 = torch.sin(x1) * torch.cos(x2)
    fact2 = torch.exp(torch.abs(1 - torch.sqrt(x1**2 + x2**2) / torch.pi))

    y = -torch.abs(fact1 * fact2)
    return y
