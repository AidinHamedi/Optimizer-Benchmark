import torch

from .norm import normalize

FUNCTION_NAME = "Rastrigin"
START_POS = torch.tensor([-8.2, 7.7])
EVAL_SIZE = ((-10, 10), (-10, 10))
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])


@normalize(0.0, 261.5675964355469)
@torch.jit.script
def rastrigin(x: torch.Tensor) -> torch.Tensor:
    """Compute the Rastrigin function.

    Args:
        x: Input tensor of shape [2] representing [x, y] coordinates.

    Returns:
        Scalar tensor with the function value.
    """
    return 10.0 * x.shape[-1] + (x.pow(2) - 10.0 * torch.cos(2.0 * torch.pi * x)).sum(
        dim=-1
    )
