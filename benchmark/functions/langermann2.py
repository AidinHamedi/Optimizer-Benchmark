import torch

from .norm import normalize

FUNCTION_NAME = "Langermann 2"
START_POS = torch.tensor([2.75, 7.38])
EVAL_SIZE = ((-1.5, 12), (-2, 12))
GLOBAL_MINIMUM_LOC = torch.tensor([[7.6557745933532715, 2.076188087463379]])

LANGERMANN_M = torch.tensor(11)
LANGERMANN_C = torch.tensor([2.5, 2.0, 1.0, 1.5, 3.0, 2.0, 2.5, 2.0, 5.0, 2.2, 1.8])
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
        [4.2, 3.6],
    ]
)


@normalize(-4.616703033447266, 5.478215217590332)
@torch.jit.script
def langermann(
    x: torch.Tensor,
    m: torch.Tensor = LANGERMANN_M,
    c: torch.Tensor = LANGERMANN_C,
    a: torch.Tensor = LANGERMANN_A,
) -> torch.Tensor:
    """Compute the Langermann function (variant 2 with 11 terms).

    Args:
        x: Input tensor of shape [2] representing [x, y] coordinates.
        m: Number of terms in the summation (default: 11).
        c: Coefficient vector of length m.
        a: Matrix of shape [m, 2] containing center points.

    Returns:
        Scalar tensor with the function value.
    """
    x_expanded = x.unsqueeze(0).expand(m, -1)  # type: ignore
    diff_sq = (x_expanded - a) ** 2
    inner = torch.sum(diff_sq, dim=1)
    terms = c * torch.exp(-inner / torch.pi) * torch.cos(torch.pi * inner)
    return torch.sum(terms)
