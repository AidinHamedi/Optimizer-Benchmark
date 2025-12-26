import torch

from .norm import normalize

FUNCTION_NAME = "Rosenbrock"
START_POS = torch.tensor([-2.0, 2.0])
EVAL_SIZE = ((-2.1, 2.1), (-1.1, 3.1))
ROSEN_A = torch.tensor(100.0)
GLOBAL_MINIMUM_LOC = torch.tensor([[1.0, 1.0]])


@normalize(0.0, 4025.572021484375)
@torch.jit.script
def rosenbrock(
    x: torch.Tensor,
    a: torch.Tensor = ROSEN_A,
) -> torch.Tensor:
    """Compute the Rosenbrock function.

    Args:
        x: Input tensor of shape [2] representing [x, y] coordinates.
        a: Steepness parameter (default: 100.0).

    Returns:
        Scalar tensor with the function value.
    """
    xi = x[:-1]
    xnext = x[1:]
    total = torch.sum(a * (xnext - xi**2) ** 2 + (xi - 1) ** 2)

    return total
