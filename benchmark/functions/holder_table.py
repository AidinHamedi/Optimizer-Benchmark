import torch

from .norm import normalize

START_POS = torch.tensor([-0.001, 0.001])
EVAL_SIZE = ((-14, 14), (-12, 12))
GLOBAL_MINIMUM_LOC = torch.tensor(
    [
        [-14.3772, -12.76168],
        [-14.3772, 12.76168],
        [14.3772, -12.76168],
        [14.3772, 12.76168],
    ]
)


@normalize(-159.3212127685547, -4.803488263860345e-05)
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
