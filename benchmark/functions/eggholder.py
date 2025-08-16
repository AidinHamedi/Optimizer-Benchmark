import torch

from .norm import normalize

START_POS = torch.tensor([0.9294921875, 0.9490234375])
EVAL_SIZE = ((-10, 10), (-10, 10))
GLOBAL_MINIMUM_LOC = torch.tensor(
    [
        [10.00000000000, 7.89515429688],
        [10.40000000000, -11.92380900000],
        [-11.54285700000, -12.00000000000],
    ]
)


@normalize(-1092.7841, 1199.1733)
@torch.jit.script
def eggholder(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Eggholder function.

    Args:
        x (torch.Tensor): A 1D tensor with two elements representing the input vector [x1, x2].

    Returns:
        torch.Tensor: A scalar tensor representing the value of the Eggholder function
            for the given input 'x'.
    """
    # To make the eval size smaller
    x1 = x[0] * 51.2
    x2 = x[1] * 51.2

    term1 = -(x2 + 47) * torch.sin(torch.sqrt(torch.abs(x2 + x1 / 2 + 47)))
    term2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47))))

    y = term1 + term2
    return y
