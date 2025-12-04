import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import torch

from .utils.executor import execute_steps
from .utils.model import Pos2D


@dataclass(frozen=True)
class ObjectiveConfig:
    """
    Configuration for the objective function scoring.

    Attributes:
        boundary_penalty (bool): If True, heavily penalizes positions outside bounds.

        # Weights (Set to 0.0 to disable specific metrics)
        final_val_weight: float = 1.6     # Weight for the final function value (log-scaled).
        final_dist_weight: float = 2.6    # Weight for final distance to global minimum.
        convergence_weight: float = 0.1   # Weight for speed of convergence.
        efficiency_weight: float = 0.2    # Weight for path efficiency (penalizes wandering).
        stagnation_weight: float = 0.5    # Weight for penalizing lack of movement (std dev).
        lucky_jump_weight: float = 1.0    # Weight for penalizing single massive steps.
        start_prox_weight: float = 10.0   # Weight for ending too close to start (Heavy penalty).
        boundary_weight: float = 16.0     # Multiplier for boundary violations.

        # Thresholds (Relative to search space diagonal)
        convergence_tol: float = 0.01      # Normalized distance to consider "converged".
        boundary_tol: float = 0.1          # Normalized distance to consider "boundary violation".
        stagnation_threshold: float = 0.01 # Min normalized std-dev to not be considered stagnant.
        lucky_jump_threshold: float = 0.05 # Max allowed single step size (as % of diagonal).
        start_prox_threshold: float = 0.12 # Distance from start to consider "no net movement".
    """

    boundary_penalty: bool = True

    final_val_weight: float = 1.6
    final_dist_weight: float = 2.6
    convergence_weight: float = 0.1
    efficiency_weight: float = 0.2
    stagnation_weight: float = 0.5
    lucky_jump_weight: float = 1.0
    start_prox_weight: float = 100.0
    boundary_weight: float = 16.0

    convergence_tol: float = 0.01
    boundary_tol: float = 0.1
    stagnation_threshold: float = 0.01
    lucky_jump_threshold: float = 0.05
    start_prox_threshold: float = 0.12


def _get_diagonal(bounds: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
    """Computes the diagonal length of the search space for normalization."""
    x_range = bounds[0][1] - bounds[0][0]
    y_range = bounds[1][1] - bounds[1][0]
    return math.sqrt(x_range**2 + y_range**2)


def _calc_boundary_violation(
    steps: torch.Tensor,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    tol: float,
) -> torch.Tensor:
    """Calculates sum of distances outside allowed bounds."""
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # steps: [2, N]
    x, y = steps[0], steps[1]

    # ReLU captures magnitude of violation (0 if inside)
    x_loss = torch.relu(x_min - x - tol) + torch.relu(x - x_max - tol)
    y_loss = torch.relu(y_min - y - tol) + torch.relu(y - y_max - tol)

    return torch.sum(x_loss + y_loss)


def _calc_path_inefficiency(
    steps: torch.Tensor, step_lengths: torch.Tensor
) -> torch.Tensor:
    """
    Calculates Path Efficiency Ratio: (Total Length / Net Displacement) - 1.
    0 means perfect straight line. Higher means wandering/oscillating.
    """
    path_len = torch.sum(step_lengths)
    displacement = torch.norm(steps[:, -1] - steps[:, 0])

    # Avoid div/0
    if displacement < 1e-6:
        return torch.tensor(0.0)

    return torch.relu((path_len / displacement) - 1.0)


def _calc_lucky_jump(step_lengths: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Penalizes if the largest single step exceeds a threshold (teleportation).
    """
    max_step = torch.max(step_lengths)

    if max_step > threshold:
        # Quadratic penalty for the excess amount
        return (max_step - threshold) ** 2

    return torch.tensor(0.0)


def _calc_stagnation(steps: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Returns penalty if the standard deviation of movement is below threshold.
    """
    # std across the path dimension
    std_dev = torch.norm(torch.std(steps, dim=1))

    if std_dev < threshold:
        # Linear penalty: Max penalty if std is 0, 0 penalty if std >= threshold
        return (threshold - std_dev) / threshold

    return torch.tensor(0.0)


def _calc_start_proximity(
    start: torch.Tensor, final: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    Penalizes if the final position is too close to the starting position
    (indicating zero net movement or circular logic).
    """
    dist = torch.norm(final - start)

    if dist < threshold:
        # Linear penalty: 1.0 if dist is 0, 0.0 if dist is threshold
        return (threshold - dist) / threshold

    return torch.tensor(0.0)


def _calc_convergence_speed(
    steps: torch.Tensor, global_min: torch.Tensor, tol: float
) -> float:
    """
    Returns the fraction of iterations spent NOT converged (0.0 to 1.0).
    Lower is better.
    """
    # steps: [2, N], global_min: [M, 2]
    # Calculate min dist to any global minimum at every step
    # We use broadcasting to calculate distance from every step to every global min
    dists = (
        torch.norm(steps.T.unsqueeze(1) - global_min.unsqueeze(0), dim=2)
        .min(dim=1)
        .values
    )

    matches = torch.nonzero(dists < tol, as_tuple=True)[0]

    if len(matches) > 0:
        first_step_idx = matches[0].item()
        total_steps = steps.shape[1] - 1
        return float(first_step_idx) / max(1, total_steps)

    return 1.0


def objective(
    criterion: Callable[[torch.Tensor], torch.Tensor],
    optimizer_maker: Callable[[Pos2D, Dict, int], Any],
    optimizer_conf: Dict[str, Any],
    start_pos: torch.Tensor,
    global_min_pos: torch.Tensor,
    bounds: Tuple[Tuple[int, int], Tuple[int, int]],
    num_iters: int,
    config: ObjectiveConfig = ObjectiveConfig(),
    debug: bool = False,
    **eval_args: Any,
) -> Tuple[float, Dict[str, float]]:
    """
    Benchmarks an optimizer trajectory and returns a weighted error score.

    Args:
        criterion: The function to minimize.
        optimizer_maker: Factory function to create the optimizer.
        optimizer_conf: Hyperparameters for the optimizer.
        start_pos: Starting coordinates [x, y].
        global_min_pos: Tensor of known global minima locations.
        bounds: Tuple of ((min_x, max_x), (min_y, max_y)).
        num_iters: Number of iterations to run.
        config: Dataclass containing weights and thresholds for scoring.
        debug: If True, prints debug info.
        **eval_args: Extra args passed to the execution loop (e.g., closure required).

    Returns:
        Tuple[float, dict]: Total error score (lower is better) and a metrics dict.
    """

    # 1. Setup Model & Optimizer
    cords = Pos2D(criterion, start_pos)
    optimizer = optimizer_maker(cords, optimizer_conf, num_iters)

    # 2. Execute Trajectory
    try:
        # steps shape: [2, num_iters + 1]
        steps = execute_steps(cords, optimizer, num_iters, **eval_args)
    except Exception as e:
        if debug:
            print(f"Error during optimization execution: {e}")
        return float("inf"), {}

    # Check for NaN/Inf which implies optimizer explosion
    if not torch.isfinite(steps).all():
        if debug:
            print("Optimizer generated NaN or Inf values.")
        return float("inf"), {"exploded_penalty": 1.0}

    # 3. Calculate Metrics
    error_sum = 0.0
    metrics = {}

    # Normalization factor
    diag = _get_diagonal(bounds)
    diag = diag if diag > 0 else 1.0

    # Pre-calculate common vector props for performance
    # diffs: vectors between steps [2, N]
    diffs = steps[:, 1:] - steps[:, :-1]
    # step_lengths: scalar length of each step [N]
    step_lengths = torch.norm(diffs, dim=0)

    # A. Final Function Value (Log-Scaled)
    final_pos = steps[:, -1]
    raw_val = criterion(final_pos).item()
    # Log1p handles scale differences (e.g., 0.001 vs 1000.0) smoothly
    val_penalty = math.log1p(max(0, raw_val)) * config.final_val_weight
    metrics["val_penalty"] = val_penalty
    error_sum += val_penalty

    # B. Final Distance to Global Minimum (Normalized)
    min_dist = torch.min(torch.norm(global_min_pos - final_pos, dim=1)).item()
    dist_penalty = (min_dist / diag) * config.final_dist_weight
    metrics["dist_penalty"] = dist_penalty
    error_sum += dist_penalty

    # C. Boundary Violations
    if config.boundary_penalty:
        violation = _calc_boundary_violation(
            steps, bounds, config.boundary_tol * diag
        ).item()
        bound_penalty = ((violation / diag) ** 4) * config.boundary_weight
        metrics["bound_penalty"] = bound_penalty
        error_sum += bound_penalty

    # D. Convergence Speed
    if config.convergence_weight > 0:
        abs_tol = config.convergence_tol * diag
        speed_ratio = _calc_convergence_speed(steps, global_min_pos, abs_tol)
        speed_penalty = speed_ratio * config.convergence_weight
        metrics["speed_penalty"] = speed_penalty
        error_sum += speed_penalty

    # E. Path Inefficiency (Wandering)
    if config.efficiency_weight > 0:
        inefficiency = _calc_path_inefficiency(steps, step_lengths).item()
        # Cap penalty to avoid skewing results on extremely erratic paths
        eff_penalty = min(inefficiency, 10.0) * config.efficiency_weight
        metrics["eff_penalty"] = eff_penalty
        error_sum += eff_penalty

    # F. Stagnation (Lack of movement variance)
    if config.stagnation_weight > 0:
        abs_stag_thresh = config.stagnation_threshold * diag
        stag_val = _calc_stagnation(steps, abs_stag_thresh).item()
        stag_penalty = stag_val * config.stagnation_weight
        metrics["stag_penalty"] = stag_penalty
        error_sum += stag_penalty

    # G. Lucky Jump (Teleportation check)
    if config.lucky_jump_weight > 0:
        abs_jump_thresh = config.lucky_jump_threshold * diag
        jump_val = _calc_lucky_jump(step_lengths, abs_jump_thresh).item()
        jump_penalty = jump_val * config.lucky_jump_weight
        metrics["jump_penalty"] = jump_penalty
        error_sum += jump_penalty

    # H. Start Proximity (Zero net movement check)
    if config.start_prox_weight > 0:
        abs_prox_thresh = config.start_prox_threshold * diag
        prox_val = _calc_start_proximity(start_pos, final_pos, abs_prox_thresh).item()
        prox_penalty = prox_val * config.start_prox_weight
        metrics["prox_penalty"] = prox_penalty
        error_sum += prox_penalty

    # Log-compress final error
    logged_error = 10 * math.log1p(error_sum) / math.log(11)

    if debug:
        print(
            f"[Objective] Raw Total: {error_sum:.4f} | Logged Total: {logged_error:.4f} | Breakdown: {metrics}"
        )

    return logged_error, metrics
