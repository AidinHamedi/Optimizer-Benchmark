import torch

from .norm import normalize

FUNCTION_NAME = "Beale"
START_POS = torch.tensor([1.0, 1.0])
EVAL_SIZE = ((-4.5, 4.5), (-4.5, 4.5))
CRITERION_OVERRIDES = {"val_scaler_root": 5}
GLOBAL_MINIMUM_LOC = torch.tensor([[3.0, 0.5]])


@normalize(4.929331043967977e-05, 383574.0625)
@torch.jit.script
def beale(
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute the Beale function.

    Args:
        x: Input tensor of shape [2] representing [x, y] coordinates.

    Returns:
        Scalar tensor with the function value.
    """
    x0 = x[..., 0]
    x1 = x[..., 1]

    term1 = 1.5 - x0 + x0 * x1
    term2 = 2.25 - x0 + x0 * x1**2
    term3 = 2.625 - x0 + x0 * x1**3

    return term1**2 + term2**2 + term3**2
