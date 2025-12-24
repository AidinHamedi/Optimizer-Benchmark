import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch


@dataclass(frozen=True)
class ObjectiveConfig:
    """Configuration for optimizer trajectory scoring.

    Attributes:
        boundary_penalty: Penalize positions outside bounds.
        final_val_weight: Weight for final function value (log-scaled).
        final_dist_weight: Weight for distance to global minimum.
        convergence_weight: Weight for convergence speed.
        efficiency_weight: Weight for path efficiency.
        lucky_jump_weight: Weight for penalizing large single steps.
        start_prox_weight: Weight for ending too close to start.
        boundary_weight: Multiplier for boundary violations.
        terrain_violation_weight: Weight for terrain violations.
        convergence_tol: Normalized distance threshold for convergence.
        boundary_tol: Normalized distance for boundary violation.
        terrain_violation_tol: Normalized distance threshold to check for terrain violation.
        terrain_violation_accuracy: Number of points to check for terrain violation.
        lucky_jump_threshold: Maximum allowed step size (fraction of diagonal).
        start_prox_threshold: Distance threshold for zero net movement.
    """

    boundary_penalty: bool = True

    final_val_weight: float = 1.8
    final_dist_weight: float = 2.6
    convergence_weight: float = 0.1
    efficiency_weight: float = 0.15
    lucky_jump_weight: float = 10.0
    start_prox_weight: float = 100.0
    boundary_weight: float = 16.0
    terrain_violation_weight: float = 14.0

    convergence_tol: float = 0.01
    boundary_tol: float = 0.08
    terrain_violation_tol: float = 0.01
    terrain_violation_accuracy: int = 7
    lucky_jump_threshold: float = 0.04
    start_prox_threshold: float = 0.12


def _get_diagonal(bounds: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
    """Compute the diagonal length of the search space."""
    x_range = bounds[0][1] - bounds[0][0]
    y_range = bounds[1][1] - bounds[1][0]
    return math.sqrt(x_range**2 + y_range**2)


def _calc_boundary_violation(
    steps: torch.Tensor,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    tol: float,
) -> torch.Tensor:
    """Compute sum of distances outside allowed bounds."""
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    x, y = steps[0], steps[1]

    x_loss = torch.relu(x_min - x - tol) + torch.relu(x - x_max - tol)
    y_loss = torch.relu(y_min - y - tol) + torch.relu(y - y_max - tol)

    return torch.sum(x_loss + y_loss)


def _calc_path_inefficiency(
    steps: torch.Tensor, step_lengths: torch.Tensor
) -> torch.Tensor:
    """Compute path inefficiency as (total length / displacement) - 1."""
    path_len = torch.sum(step_lengths)
    displacement = torch.norm(steps[:, -1] - steps[:, 0])

    if displacement < 1e-6:
        return torch.tensor(0.0)

    return torch.relu((path_len / displacement) - 1.0)


def _calc_lucky_jump(step_lengths: torch.Tensor, threshold: float) -> torch.Tensor:
    """Compute penalty for steps exceeding the threshold."""
    max_step = torch.max(step_lengths)

    if max_step > threshold:
        return (max_step - threshold) ** 2

    return torch.tensor(0.0)


def _calc_start_proximity(
    start: torch.Tensor, final: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Compute penalty for ending too close to the start position."""
    dist = torch.norm(final - start)

    if dist < threshold:
        return (threshold - dist) / threshold

    return torch.tensor(0.0)


def _calc_convergence_speed(
    steps: torch.Tensor, global_min: torch.Tensor, tol: float
) -> float:
    """Compute fraction of iterations spent not converged (0.0 to 1.0)."""
    # Distance from each step to nearest global minimum using broadcasting
    # steps: [2, N], global_min: [M, 2] -> dists: [N]
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


def _batch_evaluate(
    criterion: Callable[[torch.Tensor], torch.Tensor], points: torch.Tensor
) -> torch.Tensor:
    """Evaluate criterion on multiple points, handling different input shapes."""
    K = points.shape[1]

    try:
        return criterion(points.T)
    except Exception:
        results = torch.zeros(K)
        for i in range(K):
            results[i] = criterion(points[:, i])
        return results


def _calc_terrain_violation(
    steps: torch.Tensor,
    criterion: Callable[[torch.Tensor], torch.Tensor],
    min_dist: float,
    accuracy: int = 1,
) -> torch.Tensor:
    """Detects tunneling by checking 'accuracy' points along step paths."""
    starts = steps[:, :-1]
    ends = steps[:, 1:]

    dists = torch.norm(ends - starts, dim=0)
    mask = dists > min_dist

    if not mask.any():
        return torch.tensor(0.0)

    sig_starts = starts[:, mask]
    sig_ends = ends[:, mask]
    sig_vecs = sig_ends - sig_starts

    total_violation = torch.tensor(0.0)

    with torch.no_grad():
        # Get baseline elevation (ceiling) once
        val_start = _batch_evaluate(criterion, sig_starts)
        val_end = _batch_evaluate(criterion, sig_ends)
        ceiling = torch.maximum(val_start, val_end)

        # Iterate through 'accuracy' evenly spaced points
        for i in range(1, accuracy + 1):
            t = i / (accuracy + 1)

            # Interpolate: P = Start + t * (End - Start)
            check_points = sig_starts + (sig_vecs * t)

            val_points = _batch_evaluate(criterion, check_points)

            # Check violation: Point > Ceiling
            violation = torch.relu(val_points - (ceiling + 1e-3))
            total_violation += torch.sum(violation)

    return total_violation / accuracy


def objective(
    steps: torch.Tensor,
    criterion: Callable[[torch.Tensor], torch.Tensor],
    start_pos: torch.Tensor,
    global_min_pos: torch.Tensor,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    mode: str,
    config: ObjectiveConfig = ObjectiveConfig(),
    debug: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """Compute a weighted error score for an optimizer trajectory.

    Args:
        steps: Tensor of shape [2, N] containing trajectory coordinates.
        criterion: The objective function being minimized.
        start_pos: Starting coordinates [x, y].
        global_min_pos: Tensor of known global minima locations.
        bounds: Search space bounds as ((min_x, max_x), (min_y, max_y)).
        mode: Scoring mode, either "train" or "eval".
        config: Scoring configuration with weights and thresholds.
        debug: Enable debug output.

    Returns:
        Tuple of (total error score, metrics breakdown dictionary).
    """
    is_train = mode == "train"
    error_sum = 0.0
    metrics = {}

    # Diagonal length used to normalize distance-based thresholds
    diag = _get_diagonal(bounds)
    diag = diag if diag > 0 else 1.0

    # Pre-compute step vectors and lengths for efficiency
    diffs = steps[:, 1:] - steps[:, :-1]
    step_lengths = torch.norm(diffs, dim=0)

    # Final function value (log-scaled)
    final_pos = steps[:, -1]
    raw_val = criterion(final_pos).item()
    val_penalty = math.log1p(max(0, raw_val)) * config.final_val_weight
    metrics["val_penalty"] = val_penalty
    error_sum += val_penalty

    # Distance to global minimum (normalized)
    if is_train:
        min_dist = torch.min(torch.norm(global_min_pos - final_pos, dim=1)).item()
        dist_penalty = (min_dist / diag) * config.final_dist_weight
        metrics["dist_penalty"] = dist_penalty
        error_sum += dist_penalty

    # Boundary violations
    if config.boundary_penalty and is_train:
        violation = _calc_boundary_violation(
            steps, bounds, config.boundary_tol * diag
        ).item()
        bound_penalty = ((violation / diag) ** 4) * config.boundary_weight
        metrics["bound_penalty"] = bound_penalty
        error_sum += bound_penalty

    # Convergence speed
    if config.convergence_weight > 0:
        abs_tol = config.convergence_tol * diag
        speed_ratio = _calc_convergence_speed(steps, global_min_pos, abs_tol)
        speed_penalty = speed_ratio * config.convergence_weight
        metrics["speed_penalty"] = speed_penalty
        error_sum += speed_penalty

    # Path inefficiency
    if config.efficiency_weight > 0:
        inefficiency = _calc_path_inefficiency(steps, step_lengths).item()
        eff_penalty = min(inefficiency, 10.0) * config.efficiency_weight
        metrics["eff_penalty"] = eff_penalty
        error_sum += eff_penalty

    # Terrain violation
    if config.terrain_violation_weight > 0 and is_train:
        tv_penalty = _calc_terrain_violation(
            steps,
            criterion,
            config.terrain_violation_tol,
            config.terrain_violation_accuracy,
        ).item()
        tv_penalty *= config.terrain_violation_weight
        metrics["terrain_violation"] = tv_penalty
        error_sum += tv_penalty

    # Lucky jump (teleportation)
    if config.lucky_jump_weight > 0 and is_train:
        abs_jump_thresh = config.lucky_jump_threshold * diag
        jump_val = _calc_lucky_jump(step_lengths, abs_jump_thresh).item()
        jump_penalty = jump_val * config.lucky_jump_weight
        metrics["jump_penalty"] = jump_penalty
        error_sum += jump_penalty

    # Start proximity (zero net movement)
    if config.start_prox_weight > 0 and is_train:
        abs_prox_thresh = config.start_prox_threshold * diag
        prox_val = _calc_start_proximity(start_pos, final_pos, abs_prox_thresh).item()
        prox_penalty = prox_val * config.start_prox_weight
        metrics["prox_penalty"] = prox_penalty
        error_sum += prox_penalty

    # Log-compress to make scores comparable across functions
    logged_error = 10 * math.log1p(error_sum) / math.log(11)

    if debug:
        print(
            f"[Objective] Mode: {mode} | Raw Total: {error_sum:.4f} | Logged Total: {logged_error:.4f} | Breakdown: {metrics}"
        )

    return logged_error, metrics
