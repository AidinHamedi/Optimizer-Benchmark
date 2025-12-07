import torch

from .norm import normalize

FUNCTION_NAME = "Lévy 13"
START_POS = torch.tensor([-9.5, -7.7])
EVAL_SIZE = ((-10, 10), (-10, 10))
GLOBAL_MINIMUM_LOC = torch.tensor([[1, 1]])


@normalize(0.0008289171382784843, 539.6259765625)
@torch.jit.script
def levy13(x: torch.Tensor) -> torch.Tensor:
    """Compute the Lévy N.13 function.

    Args:
        x: Input tensor of shape [2] representing [x, y] coordinates.

    Returns:
        Scalar tensor with the function value.
    """
    x1, x2 = x[..., 0], x[..., 1]

    term1 = torch.sin(3 * torch.pi * x1) ** 2
    term2 = (x1 - 1) ** 2 * (1 + torch.sin(3 * torch.pi * x2) ** 2)
    term3 = (x2 - 1) ** 2 * (1 + torch.sin(2 * torch.pi * x2) ** 2)

    return term1 + term2 + term3
