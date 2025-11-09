import torch

from .norm import normalize

FUNCTION_NAME = "Cross-in-Tray"
START_POS = torch.tensor([6.6, 8.3])
EVAL_SIZE = ((-10, 10), (-10, 10))

FUNC_SCALE = 5

GLOBAL_MINIMUM_LOC = torch.tensor(
    [
        [-0.26586073637008667, -0.26586073637008667],
        [-0.26586073637008667, 0.26586073637008667],
        [0.26586073637008667, -0.26586073637008667],
        [0.26586073637008667, 0.26586073637008667],
    ]
)


@normalize(-20625.240234375, -806.4556884765625)
@torch.jit.script
def cross_in_tray(
    x: torch.Tensor,
    scale: float = FUNC_SCALE,
) -> torch.Tensor:
    """Computes the Cross-in-tray function.

    Args:
        x (torch.Tensor): A tensor with last dimension size 2, representing [x, y].
        scale (torch.Tensor): Scale factor for [x, y].

    Returns:
        torch.Tensor: Scalar tensor with the Cross-in-tray function value.
    """
    x1 = x[..., 0].type(torch.float64, non_blocking=True) * scale
    x2 = x[..., 1].type(torch.float64, non_blocking=True) * scale

    fact1 = torch.sin(x1) * torch.sin(x2)
    r = torch.sqrt(x1 * x1 + x2 * x2)
    fact2 = torch.exp(torch.abs(100.0 - r / torch.pi))

    y = -torch.pow(torch.abs(fact1 * fact2) + 1.0, 0.1)
    return y.type(x.dtype, non_blocking=True)
