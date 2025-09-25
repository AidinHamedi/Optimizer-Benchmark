import torch

from .norm import normalize

FUNCTION_NAME = "Rosenbrock"
START_POS = torch.tensor([-2.0, 2.0])
EVAL_SIZE = ((-2.1, 2.1), (-1.1, 3.1))
ROSEN_A = torch.tensor(100.0)
GLOBAL_MINIMUM_LOC = torch.tensor([[1.0, 1.0]])


@normalize(0.0032, 4855.7202)
@torch.jit.script
def rosenbrock(
    x: torch.Tensor,
    a: torch.Tensor = ROSEN_A,
) -> torch.Tensor:
    """
    Computes the Rosenbrock function.

    Args:
        x (torch.Tensor): A 1D tensor representing the input vector.
        a (torch.Tensor): The Rosenbrock constant (default = 100.0).

    Returns:
        torch.Tensor: Scalar tensor with the Rosenbrock function value.
    """
    d = x.numel()
    total = torch.tensor(0.0, dtype=x.dtype)

    for i in range(d - 1):
        xi = x[i]
        xnext = x[i + 1]
        term = a * (xnext - xi**2) ** 2 + (xi - 1) ** 2
        total = total + term

    return total
