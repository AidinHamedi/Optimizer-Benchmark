import torch

START_POS = torch.tensor([-9.5, -7.7])
EVAL_SIZE = ((-10, 10), (-10, 10))
GLOBAL_MINIMUM_LOC = torch.tensor([[1, 1]])


@torch.jit.script
def levy(x: torch.Tensor) -> torch.Tensor:
    """Computes the Lévy function.

    Args:
        x (torch.Tensor): A tensor representing the input vector [x1, x2, ..., xd].
            - If shape is (d,), returns a 0-D scalar.
            - If shape is (..., d), computes elementwise over leading batch dims.

    Returns:
        torch.Tensor: The Lévy function value(s). Scalar for (d,), or shape (...) for (..., d).
    """
    w = 1.0 + (x - 1.0) / 4.0
    term1 = torch.sin(torch.pi * w[..., 0]) ** 2

    wi = w[..., :-1]
    mid = (wi - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(torch.pi * wi + 1.0) ** 2)
    term2 = torch.sum(mid, dim=-1)

    wd = w[..., -1]
    term3 = (wd - 1.0) ** 2 * (1.0 + torch.sin(2.0 * torch.pi * wd) ** 2)

    return term1 + term2 + term3
