import torch

from .norm import normalize

FUNCTION_NAME = "Langermann 2"
START_POS = torch.tensor([3.8, 10])
EVAL_SIZE = ((-1.5, 12), (-2, 12))
LANGERMANN_M = torch.tensor(10)
LANGERMANN_C = torch.tensor([2.5, 2.0, 1.0, 1.5, 3.0, 2.0, 2.5, 2.0, 5.0, 2.2])
LANGERMANN_A = torch.tensor(
    [
        [3.0, 5.0],
        [5.0, 2.0],
        [2.0, 1.0],
        [1.0, 4.0],
        [7.0, 9.0],
        [9.0, 1.0],
        [6.0, 6.0],
        [4.5, 8.0],
        [8.0, 3.0],
        [2.5, 7.5],
    ]
)
GLOBAL_MINIMUM_LOC = torch.tensor([[7.6557745933532715, 2.076188087463379]])


@normalize(-4.629549026489258, 5.470799446105957)
@torch.jit.script
def langermann(
    x: torch.Tensor,
    m: torch.Tensor = LANGERMANN_M,
    c: torch.Tensor = LANGERMANN_C,
    a: torch.Tensor = LANGERMANN_A,
) -> torch.Tensor:
    """
    Computes the Langermann function.

    Args:
        x (torch.Tensor): A 1D tensor with two elements representing [x, y].
        m, c, a (torch.Tensor): Constants for the Langermann function.

    Returns:
        torch.Tensor: Scalar tensor with the Langermann function value.
    """
    x_expanded = x.unsqueeze(0).expand(m, -1)  # type: ignore
    diff_sq = (x_expanded - a) ** 2
    inner = torch.sum(diff_sq, dim=1)
    terms = c * torch.exp(-inner / torch.pi) * torch.cos(torch.pi * inner)
    return torch.sum(terms)
