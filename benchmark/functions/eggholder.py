import torch

from .norm import normalize

FUNCTION_NAME = "EggHolder"
START_POS = torch.tensor([6, -5])
EVAL_SIZE = ((-13, 13), (-13, 13))
GLOBAL_MINIMUM_LOC = torch.tensor(
    [[10.0, 7.8], [10.4, -11.9], [-11.5, -12.0], [-11.5, 4.8]]
)

FUNC_SCALE = 51.2


@normalize(-1454.9971923828125, 1296.7808837890625)
@torch.jit.script
def eggholder(
    x: torch.Tensor,
    scale: float = FUNC_SCALE,
) -> torch.Tensor:
    """Compute the Eggholder function.

    Args:
        x: Input tensor of shape [2] representing [x, y] coordinates.
        scale: Scaling factor applied to input (default: 51.2).

    Returns:
        Scalar tensor with the function value.
    """
    x = x * scale
    x1, x2 = x[0], x[1]

    term1 = -(x2 + 47) * torch.sin(torch.sqrt(torch.abs(x2 + x1 / 2 + 47)))
    term2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47))))

    return term1 + term2
