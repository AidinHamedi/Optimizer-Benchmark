import torch

from .norm import normalize

FUNCTION_NAME = "EggHolder"
START_POS = torch.tensor([0.9294921875, 0.9490234375])
EVAL_SIZE = ((-13, 13), (-13, 13))

FUNC_SCALE = 51.2

GLOBAL_MINIMUM_LOC = torch.tensor(
    [[10.0, 7.8], [10.4, -11.9], [-11.5, -12.0], [-11.5, 4.8]]
)


@normalize(-1454.9971923828125, 1296.7808837890625)
@torch.jit.script
def eggholder(
    x: torch.Tensor,
    scale: float = FUNC_SCALE,
) -> torch.Tensor:
    """
    Computes the Eggholder function.

    Args:
        x (torch.Tensor): A tensor with last dimension size 2, representing [x, y].

    Returns:
        torch.Tensor: Scalar tensor with the Eggholder function value.
    """
    x = x * scale
    x1, x2 = x[0], x[1]

    term1 = -(x2 + 47) * torch.sin(torch.sqrt(torch.abs(x2 + x1 / 2 + 47)))
    term2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47))))

    return term1 + term2
