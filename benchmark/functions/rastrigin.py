import torch

START_POS = torch.tensor([-8.2, 7.7])
EVAL_SIZE = ((-10, 10), (-10, 10))
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])


@torch.jit.script
def rastrigin(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Rastrigin function.

    Args:
        x (torch.Tensor): A tensor of shape (..., d) representing the input vector(s),
            where 'd' is the number of dimensions.

    Returns:
        torch.Tensor: A scalar tensor representing the value of the Rastrigin function
            for the given input 'x'.
    """
    return 10.0 * x.shape[-1] + (x.pow(2) - 10.0 * torch.cos(2.0 * torch.pi * x)).sum(
        dim=-1
    )
