import torch

from .norm import normalize

FUNCTION_NAME = "GradientLabyrinth"
START_POS = torch.tensor([-12.2, 14.0])
EVAL_SIZE = ((-16, 16), (-16, 16))
CRITERION_OVERRIDES = {"val_scaler_root": 5}
GLOBAL_MINIMUM_LOC = torch.tensor([[0.0, 0.0]])

WALL_STEEPNESS = torch.tensor(100.0)
GLOBAL_TREND = torch.tensor(0.05)
TRAP_DEPTH = torch.tensor(2.0)
TRAP_FREQ = torch.tensor(4.0)
THETA = torch.tensor(torch.pi / 4.0)


@normalize(0.010224738158285618, 24200.607421875)
@torch.jit.script
def gradient_labyrinth(
    x: torch.Tensor,
    wall: torch.Tensor = WALL_STEEPNESS,
    trend: torch.Tensor = GLOBAL_TREND,
    depth: torch.Tensor = TRAP_DEPTH,
    freq: torch.Tensor = TRAP_FREQ,
    theta: torch.Tensor = THETA,
) -> torch.Tensor:
    """Compute the Gradient Labyrinth function. (Ai Generated)

    A twisted, rotated, and shattered landscape. The global minimum lies at [0,0]
    inside a winding sine-wave valley. The valley floor is corrugated with local
    minima, and the valley itself is rotated 45 degrees, making the variables
    highly dependent (non-separable).

    Args:
        x: Input tensor of shape [2].
        wall: Coefficient for the valley wall steepness (default: 100.0).
        trend: Coefficient for the weak global quadratic bias (default: 0.05).
        depth: Amplitude of the cosine traps (default: 2.0).
        freq: Frequency of the cosine traps (default: 4.0).
        theta: Rotation angle of the coordinate system (default: pi/4).

    Returns:
        Scalar tensor with the function value.
    """
    # 1. Coordinate Rotation
    # Mixing x and y makes coordinate-wise optimization (like basic SGD) harder.
    u = x[0] * torch.cos(theta) - x[1] * torch.sin(theta)
    v = x[0] * torch.sin(theta) + x[1] * torch.cos(theta)

    # 2. The Manifold (Twisted Valley)
    # Instead of a simple parabola y=x^2, we force v to follow sin(u).
    # "wall" makes the sides of this path extremely steep.
    manifold_dist = v - torch.sin(u)
    valley = wall * manifold_dist**2

    # 3. Global Trend
    # A very weak pull along the U-axis ensures we eventually go to 0,
    # but the gradient is tiny compared to the walls.
    longitudinal_pull = trend * u**2

    # 4. The Traps (Shattered Floor)
    # We add cosine noise. We use (1 - cos) to ensure the noise is always positive
    # (creating bumps/holes) and that the minimum at (0,0) remains exactly 0.
    # We multiply separate cosines to create a grid of peaks and valleys.
    bumps = depth * (1.0 - torch.cos(freq * u) * torch.cos(freq * v))

    return valley + longitudinal_pull + bumps
