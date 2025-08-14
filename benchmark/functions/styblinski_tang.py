import torch

START_POS = torch.tensor([4.45, 0.13])
EVAL_SIZE = ((-5, 5), (-5, 5))
GLOBAL_MINIMUM_LOC = torch.tensor([[-2.903534, -2.903534]])


@torch.jit.script
def stybtang(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Styblinski–Tang function.

    Args:
        x (torch.Tensor): A 1D tensor of arbitrary dimension representing the input vector [x1, x2, ..., xd].

    Returns:
        torch.Tensor: A scalar tensor representing the value of the Styblinski–Tang function
            for the given input 'x'.
    """
    sum_terms = torch.sum(x**4 - 16 * x**2 + 5 * x)
    y = sum_terms / 2.0
    return y
