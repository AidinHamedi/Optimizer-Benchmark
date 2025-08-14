import torch

START_POS = torch.tensor([15.59, 15.59])
EVAL_SIZE = ((-20, 20), (-20, 20))
GLOBAL_MINIMUM_LOC = torch.tensor(
    [[1.3491, 1.3491], [-1.3491, 1.3491], [1.3491, -1.3491], [-1.3491, -1.3491]]
)


@torch.jit.script
def cross_in_tray(x: torch.Tensor) -> torch.Tensor:
    """Computes the Cross-in-tray function.

    Args:
        x (torch.Tensor): A tensor whose last dimension has size 2, representing [x1, x2].
            Supports arbitrary leading batch dimensions.

    Returns:
        torch.Tensor: The function value(s). If x has shape (2,), returns a 0-D scalar tensor.
            For batched inputs (..., 2), returns a tensor of shape (...).
    """
    x1 = x[..., 0]
    x2 = x[..., 1]

    fact1 = torch.sin(x1) * torch.sin(x2)
    r = torch.sqrt(x1 * x1 + x2 * x2)
    fact2 = torch.exp(torch.abs(100.0 - r / torch.pi))

    y = -1e-4 * torch.pow(torch.abs(fact1 * fact2) + 1.0, 0.1)
    return y
