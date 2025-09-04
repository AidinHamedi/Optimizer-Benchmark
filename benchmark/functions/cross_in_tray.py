import torch

from .norm import normalize

START_POS = torch.tensor([16.6, 21.3])
EVAL_SIZE = ((-22, 22), (-22, 22))
GLOBAL_MINIMUM_LOC = torch.tensor(
    [[1.3491, 1.3491], [-1.3491, 1.3491], [1.3491, -1.3491], [-1.3491, -1.3491]]
)


@normalize(-2.0626, -0.3973)
@torch.jit.script
def cross_in_tray(x: torch.Tensor) -> torch.Tensor:
    """Computes the Cross-in-tray function.

    Args:
        x (torch.Tensor): A tensor with last dimension size 2, representing [x1, x2].

    Returns:
        torch.Tensor: Scalar tensor with the Cross-in-tray function value.
    """
    x1 = x[..., 0].type(torch.float64, non_blocking=True)
    x2 = x[..., 1].type(torch.float64, non_blocking=True)

    fact1 = torch.sin(x1) * torch.sin(x2)
    r = torch.sqrt(x1 * x1 + x2 * x2)
    fact2 = torch.exp(torch.abs(100.0 - r / torch.pi))

    y = -1e-4 * torch.pow(torch.abs(fact1 * fact2) + 1.0, 0.1)
    return y.type(x.dtype, non_blocking=True)
