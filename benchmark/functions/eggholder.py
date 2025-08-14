import torch

START_POS = torch.tensor([47.59, 47.59])
EVAL_SIZE = ((-512, 512), (-512, 512))
GLOBAL_MINIMUM_LOC = torch.tensor([[512.0, 404.2319]])


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
    x1 = x[0]
    x2 = x[1]

    term1 = -(x2 + 47) * torch.sin(torch.sqrt(torch.abs(x2 + x1 / 2 + 47)))
    term2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47))))

    y = term1 + term2
    return y
