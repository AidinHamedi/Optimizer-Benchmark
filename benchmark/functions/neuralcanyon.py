import torch

from .norm import normalize

FUNCTION_NAME = "NeuralCanyon"
START_POS = torch.tensor([-4.5, 3.5])
EVAL_SIZE = ((-8, 8), (-8, 8))
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])

CANYON_WALL = torch.tensor(30.0)
GLOBAL_BIAS = torch.tensor(0.005)
NOISE_AMP = torch.tensor(1.8)
NOISE_FREQ = torch.tensor(25.0)


@normalize(-0.7994629144668579, 2888.94384765625)
@torch.jit.script
def neural_canyon(
    x: torch.Tensor,
    wall: torch.Tensor = CANYON_WALL,
    bias: torch.Tensor = GLOBAL_BIAS,
    amp: torch.Tensor = NOISE_AMP,
    freq: torch.Tensor = NOISE_FREQ,
) -> torch.Tensor:
    """Compute the Neural Canyon function. (Ai Generated)

    A landscape that mimics deep learning loss surfaces: a narrow, curving
    valley (manifold) corrupted by high-frequency noise and flattened gradients.

    Args:
        x: Input tensor of shape [2] representing [x, y] coordinates.
        wall: Steepness of the valley walls (default: 30.0).
        bias: Global quadratic regularization strength (default: 0.008).
        amp: Amplitude of the sinusoidal noise traps (default: 1.4).
        freq: Frequency of the local traps (default: 25.0).

    Returns:
        Scalar tensor with the function value.
    """
    x_coord = x[0]
    y_coord = x[1]

    # 1. The Manifold (A twisted valley following a tanh curve)
    # This creates a narrow path that is hard to navigate (ill-conditioned).
    # The tanh mimics saturation/vanishing gradients at extremities.
    manifold_path = torch.tanh(x_coord)
    valley_term = wall * (y_coord - manifold_path) ** 2

    # 2. Regularization (Weak global pull)
    # Prevents the optimizer from drifting to infinity along the flat tanh tails.
    reg_term = bias * (x_coord**2 + y_coord**2)

    # 3. The Noise (Local Minima)
    # "Egg-crate" interference pattern that traps optimizers in sub-optimal spots.
    # We square the sine to ensure the noise is strictly positive (bumps),
    # then subtract to make holes, ensuring (0,0) remains the deep global min.
    noise_term = (
        -amp
        * torch.exp(-0.1 * (x_coord**2 + y_coord**2))
        * torch.cos(freq * x_coord)
        * torch.cos(freq * y_coord)
    )

    # Combine terms
    return valley_term + reg_term + noise_term
