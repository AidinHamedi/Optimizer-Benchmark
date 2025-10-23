import torch

from .norm import normalize

FUNCTION_NAME = "Lévy"
START_POS = torch.tensor([-9.5, -7.7])
EVAL_SIZE = ((-10, 10), (-10, 10))
GLOBAL_MINIMUM_LOC = torch.tensor([[1, 1]])


@normalize(1.0799613846756984e-05, 103.99503326416016)
@torch.jit.script
def levy(x: torch.Tensor) -> torch.Tensor:
    """Computes the Lévy function.

    Args:
        x (torch.Tensor): A tensor with last dimension size 2, representing [x, y].

    Returns:
        torch.Tensor: Scalar tensor with the Lévy function value.
    """
    w = 1.0 + (x - 1.0) / 4.0
    term1 = torch.sin(torch.pi * w[..., 0]) ** 2

    wi = w[..., :-1]
    mid = (wi - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(torch.pi * wi + 1.0) ** 2)
    term2 = torch.sum(mid, dim=-1)

    wd = w[..., -1]
    term3 = (wd - 1.0) ** 2 * (1.0 + torch.sin(2.0 * torch.pi * wd) ** 2)

    return term1 + term2 + term3
