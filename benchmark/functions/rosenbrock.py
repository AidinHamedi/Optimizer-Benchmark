import torch

from .norm import normalize

FUNCTION_NAME = "Rosenbrock"
START_POS = torch.tensor([-2.0, 2.0])
EVAL_SIZE = ((-2.1, 2.1), (-1.1, 3.1))
ROSEN_A = torch.tensor(100.0)
GLOBAL_MINIMUM_LOC = torch.tensor([[1.0, 1.0]])


@normalize(0.0007723920280113816, 4025.572021484375)
@torch.jit.script
def rosenbrock(
    x: torch.Tensor,
    a: torch.Tensor = ROSEN_A,
) -> torch.Tensor:
    """
    Computes the Rosenbrock function.

    Args:
        x (torch.Tensor): A tensor with last dimension size 2, representing [x, y].
        a (torch.Tensor): The Rosenbrock constant (default = 100.0).

    Returns:
        torch.Tensor: Scalar tensor with the Rosenbrock function value.
    """
    xi = x[:-1]
    xnext = x[1:]
    total = torch.sum(a * (xnext - xi**2) ** 2 + (xi - 1) ** 2)

    return total
