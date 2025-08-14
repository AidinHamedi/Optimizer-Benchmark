import torch

START_POS = torch.tensor([41, 41])
EVAL_SIZE = ((-50, 50), (-50, 50))
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])


@torch.jit.script
def schaffer2(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Schaffer function N. 2.

    Args:
        x (torch.Tensor): A 1D tensor with two elements representing the input vector [x1, x2].

    Returns:
        torch.Tensor: A scalar tensor representing the value of the Schaffer function
            for the given input 'x'.
    """
    x1 = x[0]
    x2 = x[1]

    fact1 = torch.sin(x1**2 - x2**2) ** 2 - 0.5
    fact2 = (1 + 0.001 * (x1**2 + x2**2)) ** 2

    y = 0.5 + fact1 / fact2
    return y
