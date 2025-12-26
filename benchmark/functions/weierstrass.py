import torch

from .norm import normalize

FUNCTION_NAME = "Weierstrass"
START_POS = torch.tensor([-12, -11])
EVAL_SIZE = ((-13, 13), (-13, 13))
GLOBAL_MINIMUM_LOC = torch.tensor([[1.2642141580581665, 1.2642141580581665]])

WEIERSTRASS_A = 0.55
WEIERSTRASS_B = 2.5
WEIERSTRASS_KMAX = 5.0
_k = torch.arange(WEIERSTRASS_KMAX + 1)
WEIERSTRASS_AK = torch.pow(WEIERSTRASS_A, _k)
WEIERSTRASS_PIBK = torch.pi * torch.pow(WEIERSTRASS_B, _k)
FUNC_SCALE = 0.07692307692


@normalize(-3.7680187225341797, 4.321251392364502)
@torch.jit.script
def weierstrass(
    x: torch.Tensor,
    ak: torch.Tensor = WEIERSTRASS_AK,
    pibk: torch.Tensor = WEIERSTRASS_PIBK,
    scale: float = FUNC_SCALE,
) -> torch.Tensor:
    """Compute the Weierstrass function.

    Args:
        x: Input tensor of shape [2] representing [x, y] coordinates.
        ak: Precomputed a^k coefficients.
        pibk: Precomputed π*b^k coefficients.
        scale: Scaling factor applied to input.

    Returns:
        Scalar tensor with the function value.
    """
    x = x * scale + 1.0

    x_expanded = x.unsqueeze(1)
    pibk_expanded = pibk.unsqueeze(0)

    cos_terms = torch.cos(x_expanded * pibk_expanded)
    inner_sums = torch.sum(ak * cos_terms, dim=1)

    return torch.sum(inner_sums)
