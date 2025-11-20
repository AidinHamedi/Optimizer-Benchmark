import torch

from .norm import normalize

FUNCTION_NAME = "Weierstrass"
START_POS = torch.tensor([-12, -11])
EVAL_SIZE = ((-13, 13), (-13, 13))
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])

WEIERSTRASS_A = 0.55
WEIERSTRASS_B = 3.0
WEIERSTRASS_KMAX = 10
_k = torch.arange(WEIERSTRASS_KMAX + 1)
WEIERSTRASS_AK = torch.pow(WEIERSTRASS_A, _k)
WEIERSTRASS_PIBK = torch.pi * torch.pow(WEIERSTRASS_B, _k)
FUNC_SCALE = 0.07692307692


@normalize(-3.850635290145874, 3.9265196323394775)
@torch.jit.script
def weierstrass(
    x: torch.Tensor,
    ak: torch.Tensor = WEIERSTRASS_AK,
    pibk: torch.Tensor = WEIERSTRASS_PIBK,
    scale: float = FUNC_SCALE,
) -> torch.Tensor:
    """
    Computes the Weierstrass function.

    Args:
        x (torch.Tensor): A 1D tensor representing the input vector.
        ak (torch.Tensor): Precomputed a^k coefficients.
        pibk (torch.Tensor): Precomputed π*b^k coefficients.

    Returns:
        torch.Tensor: Scalar tensor with the Weierstrass function value.
    """
    x = x * scale + 1.0

    x_expanded = x.unsqueeze(1)
    pibk_expanded = pibk.unsqueeze(0)

    cos_terms = torch.cos(x_expanded * pibk_expanded)
    inner_sums = torch.sum(ak * cos_terms, dim=1)

    return torch.sum(inner_sums)
