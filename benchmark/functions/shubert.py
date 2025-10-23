import torch

from .norm import normalize

FUNCTION_NAME = "Shubert"
START_POS = torch.tensor([1.81, 1.82])
EVAL_SIZE = ((-10, 10), (-10, 10))
GLOBAL_MINIMUM_LOC = torch.tensor(
    [
        [-7.7778, -7.1717],
        [-7.7778, -0.9091],
        [-7.7778, 5.3535],
        [-6.9697, -7.7778],
        [-6.9697, -1.5152],
        [-6.9697, 4.7475],
        [-1.5152, -7.1717],
        [-1.5152, -0.9091],
        [-1.5152, 5.3535],
        [-0.7071, -7.7778],
        [-0.7071, -1.5152],
        [-0.7071, 4.7475],
        [4.7475, -7.1717],
        [4.7475, -0.9091],
        [4.7475, 5.5556],
        [5.5556, -7.7778],
        [5.5556, -1.5152],
        [5.5556, 4.7475],
    ]
)


@normalize(-186.72389221191406, 210.48094177246094)
@torch.jit.script
def shubert(x: torch.Tensor) -> torch.Tensor:
    """Computes the Shubert function.

    Args:
        x (torch.Tensor): A tensor with last dimension size 2, representing [x, y].

    Returns:
        torch.Tensor: Scalar tensor with the Shubert function value.
    """
    i = torch.arange(1, 6, dtype=x.dtype, device=x.device)

    x1 = x[..., 0].unsqueeze(-1)
    x2 = x[..., 1].unsqueeze(-1)

    sum1 = torch.sum(i * torch.cos((i + 1) * x1 + i), dim=-1)
    sum2 = torch.sum(i * torch.cos((i + 1) * x2 + i), dim=-1)

    return sum1 * sum2
