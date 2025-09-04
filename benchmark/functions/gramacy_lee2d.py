import torch

from .norm import normalize

START_POS = torch.tensor([2.2, 2.45])
EVAL_SIZE = ((0.5, 2.5), (0.5, 2.5))
GLOBAL_MINIMUM_LOC = torch.tensor([[0.55, 0.55]])


@torch.jit.script
def _gramacy_lee_1d(val: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gramacy & Lee (1D) function.

    Args:
        val (torch.Tensor): A tensor with last dimension size 1.

    Returns:
        torch.Tensor: Scalar tensor with the function value.
    """
    eps = 1e-8
    term1 = torch.where(
        torch.abs(val) < eps,
        torch.tensor(1, device=val.device, dtype=val.dtype),
        torch.sin(10 * torch.pi * val) / (2 * val),
    )
    term2 = (val - 1) ** 4
    return term1 + term2


@normalize(-1.7335622310638428, 10.124999046325684)
@torch.jit.script
def gl2d(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gramacy & Lee (2D) function as f(x) + f(y).

    Args:
        x (torch.Tensor): A tensor with last dimension size 2, representing [x, y].

    Returns:
        torch.Tensor: Scalar tensor with the function value.
    """
    return _gramacy_lee_1d(x[..., 0]) + _gramacy_lee_1d(x[..., 1])
