import torch

from .norm import normalize

FUNCTION_NAME = "Griewank"
START_POS = torch.tensor([-57.0, -42.6])
EVAL_SIZE = ((-60, 60), (-60, 60))
CRITERION_OVERRIDES = {"val_scaler_root": 4}
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])

FUNC_SCALE = 10


@normalize(-0.9906126856803894, 217.95753479003906)
@torch.jit.script
def griewank(
    x: torch.Tensor,
    scale: float = FUNC_SCALE,
) -> torch.Tensor:
    """Compute the Griewank function.

    Args:
        x: Input tensor of shape [2] representing [x, y] coordinates.
        scale: Scaling factor applied to input (default: 10).

    Returns:
        Scalar tensor with the function value.
    """
    d = x.shape[-1]

    x = x * scale

    sum_term = torch.sum(x**2, dim=-1) / 4000.0
    i = torch.arange(1, d + 1, dtype=x.dtype, device=x.device)
    prod_term = torch.prod(torch.cos(x / torch.sqrt(i)), dim=-1)

    return sum_term - prod_term
