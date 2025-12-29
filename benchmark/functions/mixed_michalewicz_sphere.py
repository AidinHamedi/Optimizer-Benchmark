import torch

from .norm import normalize

FUNCTION_NAME = "MixedMichalewiczSphere"
START_POS = torch.tensor([9.0, 9.0])
EVAL_SIZE = ((-10.0, 10.0), (-10.0, 10.0))
GLOBAL_MINIMUM_LOC = torch.tensor([[2.1268880367279053, 1.5619332790374756]])

M = 2.0
ALPHA = 0.95
SPHERE_SCALE = 0.6


@normalize(-1.5201987028121948, 8.873723983764648)
@torch.jit.script
def mixed_michalewicz_sphere(
    x: torch.Tensor,
    m: float = M,
    alpha: float = ALPHA,
    sphere_scale: float = SPHERE_SCALE,
) -> torch.Tensor:
    """Compute the Mixed Michalewicz + Sphere function.

    Args:
        x: Input tensor of shape [2] representing [x, y] coordinates.
        m: Michalewicz steepness parameter (default: 2.0).
        alpha: Mixing coefficient between Michalewicz and Sphere (default: 0.95).
        sphere_scale: Scaling factor for the Sphere component (default: 0.6).

    Returns:
        Scalar tensor with the function value.
    """
    sphere_val = torch.sum(x * x)

    n = x.size(0)
    i = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)

    sin_x = torch.sin(x)
    inner = torch.sin(i * (x * x) / torch.pi)
    term = sin_x * (inner ** (2.0 * m))
    mich_sum = torch.sum(term)

    return alpha * (-mich_sum) + (1.0 - alpha) * (sphere_scale * sphere_val)
