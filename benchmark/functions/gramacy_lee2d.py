import torch

from .norm import normalize

FUNCTION_NAME = "Gramacy & Lee 2D"
START_POS = torch.tensor([1.8, 2.48])
EVAL_SIZE = ((-0.8, 2.5), (-0.8, 2.5))
GLOBAL_MINIMUM_LOC = torch.tensor(
    [
        [0.14166, 0.14166],
    ]
)


@torch.jit.script
def _gramacy_lee_1d(val: torch.Tensor) -> torch.Tensor:
    """Compute the 1D Gramacy & Lee function."""
    eps = 1e-8
    term1 = torch.where(
        torch.abs(val) < eps,
        torch.tensor(1, device=val.device, dtype=val.dtype),
        torch.sin(10 * torch.pi * val) / (2 * val),
    )
    term2 = (val - 1) ** 4
    return term1 + term2


@normalize(-5.706794738769531, 33.4173583984375)
@torch.jit.script
def gl2d(x: torch.Tensor) -> torch.Tensor:
    """Compute the Gramacy & Lee 2D function as f(x) + f(y).

    Args:
        x: Input tensor of shape [2] representing [x, y] coordinates.

    Returns:
        Scalar tensor with the function value.
    """
    return _gramacy_lee_1d(x[..., 0]) + _gramacy_lee_1d(x[..., 1])
