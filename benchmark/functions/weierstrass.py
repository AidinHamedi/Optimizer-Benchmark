import torch

from .norm import normalize

FUNCTION_NAME = "Weierstrass"
START_POS = torch.tensor([-12, -11])
EVAL_SIZE = ((-13, 13), (-13, 13))
WEIERSTRASS_A = torch.tensor(0.5)
WEIERSTRASS_B = torch.tensor(3.0)
WEIERSTRASS_KMAX = torch.tensor(20)

_kmax = WEIERSTRASS_KMAX.int().item()
_k = torch.arange(_kmax + 1)
WEIERSTRASS_AK = torch.pow(WEIERSTRASS_A, _k)
WEIERSTRASS_PIBK = torch.pi * torch.pow(WEIERSTRASS_B, _k)

GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])


@normalize(0.149155855178833, 7.9263105392456055)
@torch.jit.script
def weierstrass(
    x: torch.Tensor,
    ak: torch.Tensor = WEIERSTRASS_AK,
    pibk: torch.Tensor = WEIERSTRASS_PIBK,
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
    x_normalized = x / 26.0
    x_scaled = 2.0 * x_normalized + 1.0

    x_expanded = x_scaled.unsqueeze(1)
    pibk_expanded = pibk.unsqueeze(0)

    cos_terms = torch.cos(x_expanded * pibk_expanded)
    inner_sums = torch.sum(ak * cos_terms, dim=1)

    return torch.sum(inner_sums)
