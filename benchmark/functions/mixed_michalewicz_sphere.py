import torch

from .norm import normalize

FUNCTION_NAME = "MixedMichalewiczSphere"
START_POS = torch.tensor([9.0, 9.0])
EVAL_SIZE = ((-10.0, 10.0), (-10.0, 10.0))
GLOBAL_MINIMUM_LOC = torch.tensor([[2.1268880367279053, 1.5619332790374756]])

M = 2.0
ALPHA = 0.95
SPHERE_SCALE = 0.6


@normalize(-1.5200809240341187, 8.869929313659668)
@torch.jit.script
def mixed_michalewicz_sphere(
    x: torch.Tensor,
    m: float = M,
    alpha: float = ALPHA,
    sphere_scale: float = SPHERE_SCALE,
) -> torch.Tensor:
    """
    Computes Mixed Michalewicz + Sphere function.

    Args:
        x (torch.Tensor): A tensor with last dimension size 2, representing [x, y].
        m (float): Michalewicz steepness parameter.
        alpha (float): mixing coefficient (0 ≤ α ≤ 1).
        sphere_scale (float): scale factor for the sphere component.

    Returns:
        torch.Tensor: Scalar tensor with the function value.
    """
    sphere_val = torch.sum(x * x)

    n = x.size(0)
    i = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)

    sin_x = torch.sin(x)
    inner = torch.sin(i * (x * x) / torch.pi)
    term = sin_x * (inner ** (2.0 * m))
    mich_sum = torch.sum(term)

    return alpha * (-mich_sum) + (1.0 - alpha) * (sphere_scale * sphere_val)
