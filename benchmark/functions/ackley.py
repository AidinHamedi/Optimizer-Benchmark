import torch

from .norm import normalize

FUNCTION_NAME = "Ackley"
START_POS = torch.tensor([7.6, 8.4])
EVAL_SIZE = ((-10, 10), (-10, 10))
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])

ACKLEY_A = torch.tensor(10.0)
ACKLEY_B = torch.tensor(0.1)
ACKLEY_C = torch.tensor(2 * torch.pi)


@normalize(-12.718281745910645, -3.8639559745788574)
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
        x (torch.Tensor): A tensor with last dimension size 2, representing [x, y].
        a, b, c (torch.Tensor): Constants for the Ackley function.

    Returns:
        torch.Tensor: Scalar tensor with the Ackley function value.
    """
    d = x.numel()
    sum1 = torch.sum(x**2)
    sum2 = torch.sum(torch.cos(c * x))

    term1 = -a * torch.exp(-b * torch.sqrt(sum1 / d))
    term2 = -torch.exp(sum2 / d)

    return term1 + term2
