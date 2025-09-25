import torch

from .norm import normalize

FUNCTION_NAME = "Rastrigin"
START_POS = torch.tensor([-8.2, 7.7])
EVAL_SIZE = ((-10, 10), (-10, 10))
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])


@normalize(0.6358, 305.3365)
@torch.jit.script
def rastrigin(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Rastrigin function.

    Args:
        x (torch.Tensor): A tensor representing the input vector.

    Returns:
        torch.Tensor: Scalar tensor with the Rastrigin function value.
    """
    return 10.0 * x.shape[-1] + (x.pow(2) - 10.0 * torch.cos(2.0 * torch.pi * x)).sum(
        dim=-1
    )
