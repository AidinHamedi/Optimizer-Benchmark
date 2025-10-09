import torch

from .norm import normalize

FUNCTION_NAME = "Griewank"
START_POS = torch.tensor([-57.0, -42.6])
EVAL_SIZE = ((-60, 60), (-60, 60))
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])


@normalize(0.029671549797058105, 218.95753479003906)
@torch.jit.script
def griewank(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Griewank function.

    Args:
        x (torch.Tensor): A 1D tensor representing the input vector.

    Returns:
        torch.Tensor: Scalar tensor with the Griewank function value.
    """
    d = x.shape[-1]

    # To make the eval size smaller
    x = x * 10

    sum_term = torch.sum(x**2, dim=-1) / 4000.0
    i = torch.arange(1, d + 1, dtype=x.dtype)
    prod_term = torch.prod(torch.cos(x / torch.sqrt(i)), dim=-1)

    return sum_term - prod_term + 1.0
