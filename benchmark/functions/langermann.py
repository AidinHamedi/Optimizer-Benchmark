import torch

START_POS = torch.tensor([4.6, 6.7])
EVAL_SIZE = ((0, 10), (0, 10))
LANGERMANN_M = torch.tensor(5)
LANGERMANN_C = torch.tensor([1.0, 2.0, 5.0, 2.0, 3.0])
LANGERMANN_A = torch.tensor(
    [[3.0, 5.0], [5.0, 2.0], [2.0, 1.0], [1.0, 4.0], [7.0, 9.0]]
)
GLOBAL_MINIMUM_LOC = torch.tensor([[2.7927, 1.6016]])


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
        x (torch.Tensor): A 1D tensor of shape (d,) representing the input vector.
        a, b, c (torch.Tensor): Constant

    Returns:
        torch.Tensor: A scalar tensor representing the Langermann function value.
    """
    x_expanded = x.unsqueeze(0).expand(m, -1)  # type: ignore
    diff_sq = (x_expanded - a) ** 2
    inner = torch.sum(diff_sq, dim=1)
    terms = c * torch.exp(-inner / torch.pi) * torch.cos(torch.pi * inner)
    return torch.sum(terms)
