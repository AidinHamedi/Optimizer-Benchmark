import torch

from .norm import normalize

START_POS = torch.tensor([7.6, 8.4])
EVAL_SIZE = ((-10, 10), (-10, 10))
ACKLEY_A = torch.tensor(10.0)
ACKLEY_B = torch.tensor(0.1)
ACKLEY_C = torch.tensor(2 * torch.pi)
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])


@normalize(-0.1247, 9.1874)
@torch.jit.script
def ackley(
    x: torch.Tensor,
    a: torch.Tensor = ACKLEY_A,
    b: torch.Tensor = ACKLEY_B,
    c: torch.Tensor = ACKLEY_C,
) -> torch.Tensor:
    """
    Computes the Ackley function.

    Args:
        x (torch.Tensor): A tensor of shape (..., d) representing the input vector(s),
            where 'd' is the number of dimensions.
        a, b, c (torch.Tensor): Constant

    Returns:
        torch.Tensor: A scalar tensor representing the value of the Ackley function
            for the given input 'x'.
    """
    d = x.numel()
    sum1 = torch.sum(x**2)
    sum2 = torch.sum(torch.cos(c * x))

    term1 = -a * torch.exp(-b * torch.sqrt(sum1 / d))
    term2 = -torch.exp(sum2 / d)

    return term1 + term2 + a + torch.e
