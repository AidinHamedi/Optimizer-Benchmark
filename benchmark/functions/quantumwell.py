import torch

from .norm import normalize

FUNCTION_NAME = "QuantumWell"
START_POS = torch.tensor([8.2, 7.5])
EVAL_SIZE = ((-10, 10), (-10, 10))
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])

QUANTUM_SCALE = torch.tensor(0.05)
QUANTUM_AMP = torch.tensor(4.0)
QUANTUM_FREQ = torch.tensor(2.5)
QUANTUM_DECAY = torch.tensor(0.15)


@normalize(-3.9932661056518555, 11.901788711547852)
@torch.jit.script
def quantum_well(
    x: torch.Tensor,
    scale: torch.Tensor = QUANTUM_SCALE,
    amp: torch.Tensor = QUANTUM_AMP,
    freq: torch.Tensor = QUANTUM_FREQ,
    decay: torch.Tensor = QUANTUM_DECAY,
) -> torch.Tensor:
    """Compute the Quantum Well function. (Ai Generated)

    A complex multi-modal landscape consisting of a global quadratic basin
    overlaid with a decaying cosine lattice. The global minimum is at [0, 0],
    protected by numerous local optima that act as barriers.

    Args:
        x: Input tensor of shape [2] representing [x, y] coordinates.
        scale: Quadratic slope coefficient (default: 0.05).
        amp: Amplitude of the cosine lattice holes (default: 4.0).
        freq: Frequency of the lattice oscillation (default: 2.5).
        decay: Exponential decay rate of the lattice amplitude (default: 0.15).

    Returns:
        Scalar tensor with the function value.
    """
    # Calculate distance from center
    sum_sq = torch.sum(x**2)
    dist = torch.sqrt(sum_sq)

    # 1. Global Quadratic Basin (Pull towards center)
    basin = scale * sum_sq

    # 2. Lattice Trap (Grid of local minima)
    # Uses product of cosines to create a grid structure
    oscillation = torch.prod(torch.cos(freq * x))

    # 3. Dampening (Lattice gets weaker further away)
    damping = torch.exp(-decay * dist)

    # Combine: Basin - (Traps * Damping)
    # We subtract because we want the 'holes' to dig into the basin
    return basin - (amp * oscillation * damping)
