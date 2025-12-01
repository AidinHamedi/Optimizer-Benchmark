import math
from typing import Callable, Dict, Tuple

import torch
from scipy.spatial import ConvexHull

from .utils.executor import execute_steps
from .utils.model import Pos2D


def objective(
    criterion: Callable,
    optimizer_maker: Callable,
    optimizer_conf: dict,
    start_pos: torch.Tensor,
    global_min_pos: torch.Tensor,
    bounds: Tuple[Tuple[int, int], Tuple[int, int]],
    num_iters: int,
    boundary_penalty: bool = True,
    average_distance_factor: float = 0.0,
    convergence_factor: float = 0.01,
    convergence_tol: float = 0.02,
    oscillation_factor: float = 1.0,
    lucky_jump_factor: float = 2.0,
    lucky_jump_threshold: float = 0.05,
    final_distance_factor: float = 1.0,
    final_value_factor: float = 1.0,
    min_movement_factor: float = 0.6,
    min_movement_threshold: float = 0.1,
    final_proximity_factor: float = 8.0,
    final_proximity_threshold: float = 0.1,
    debug: bool = False,
    **eval_args,
) -> Tuple[float, dict]:
    """
    Evaluates an optimization trajectory and returns a scalar error.

    Args:
        criterion (Callable): Function to minimize (maps [x, y] -> scalar).
        optimizer_maker (Callable): Factory that creates an optimizer instance.
        optimizer_conf (dict): Optimizer configuration parameters.
        start_pos (torch.Tensor): Starting position (shape: [2]).
        global_min_pos (torch.Tensor): One or more known global minima ([N, 2]).
        bounds (Tuple[Tuple[int, int], Tuple[int, int]]): Allowed (x, y) ranges.
        num_iters (int): Number of optimization steps.
        boundary_penalty (bool, optional): Penalize positions outside bounds.
        average_distance_factor (float, optional): Weight for trajectory wandering penalty.
        convergence_factor (float, optional): Weight for convergence speed penalty. 0 disables.
        convergence_tol (float, optional): Relative distance to a minimum to be considered "converged".
        oscillation_factor (float, optional): Weight for oscillation penalty. 0 disables.
        lucky_jump_factor (float, optional): Weight for very large step penalty. 0 disables.
        lucky_jump_threshold (float, optional): Relative step size considered "too large".
        final_distance_factor (float, optional): Weight for final distance to global minima. 0 disables.
        final_value_factor (float, optional): Weight for final function value.
        min_movement_factor (float, optional): Weight for insufficient-movement penalty. 0 disables.
        min_movement_threshold (float, optional): Fraction of max side considered too small movement.
        final_proximity_factor (float, optional): Weight for final-position-close-to-start penalty. 0 disables.
        final_proximity_threshold (float, optional): Fraction of max side considered too close.
        debug (bool, optional): Print debug information.
        **eval_args: Extra arguments for the step execution function.

    Returns:
        Tuple[float, Dict[str, float]]: Total error (lower is better) and metrics.
    """
    cords = Pos2D(criterion, start_pos)
    optimizer = optimizer_maker(cords, optimizer_conf, num_iters)

    try:
        steps = execute_steps(cords, optimizer, num_iters, **eval_args)
    except Exception as e:
        print(f"Error during optimization: {e}")
        return float("inf"), {}

    final_pos = steps[:, -1]
    error: float = 0.0
    metrics: Dict[str, float] = {}

    def error_scaler(x):
        return math.sqrt(x) + x / 3

    # Normalization
    # To make the error independent of the function's scale, we normalize all
    # distance-based metrics by the diagonal of the search space.
    ranges = torch.tensor(
        [bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]],
        dtype=torch.float32,
    )
    max_side = ranges.max().item()
    search_space_diag = torch.norm(ranges).item()

    # 0. Function value at final position.
    final_value = max(criterion(final_pos).item(), 0)
    contrib = error_scaler(final_value * final_value_factor)
    metrics["final loss"] = contrib
    error += contrib

    # 1. Boundary penalty.
    if boundary_penalty:
        violation = (
            torch.clamp(bounds[0][0] - steps[0], min=0).max()
            + torch.clamp(steps[0] - bounds[0][1], min=0).max()
            + torch.clamp(bounds[1][0] - steps[1], min=0).max()
            + torch.clamp(steps[1] - bounds[1][1], min=0).max()
        )
        normalized_violation = violation.item() / search_space_diag
        contrib = (normalized_violation**2) * 40.0  # Weighted heavily
        metrics["boundary violation"] = contrib
        error += contrib

        final_violation = (
            max(bounds[0][0] - final_pos[0].item(), 0)
            + max(final_pos[0].item() - bounds[0][1], 0)
            + max(bounds[1][0] - final_pos[1].item(), 0)
            + max(final_pos[1].item() - bounds[1][1], 0)
        )
        if final_violation > 0:
            contrib = (final_violation / search_space_diag) * 120
            metrics["final position out-of-bounds"] = contrib
            error += contrib

    # 2. Final distance to the nearest known global minimum.
    final_dist = max(
        torch.min(torch.norm(global_min_pos - final_pos, dim=1)).item() - 0.04, 0
    )
    normalized_final_dist = final_dist / search_space_diag
    contrib = error_scaler(normalized_final_dist * final_distance_factor)
    metrics["final distance to global minimum"] = contrib
    error += contrib

    # 3. Average trajectory distance from the final point (wandering).
    if average_distance_factor > 0.0:
        avg_dist = torch.norm(steps.T - final_pos[None, :], dim=1).mean().item()
        normalized_avg_dist = avg_dist / search_space_diag
        contrib = error_scaler(normalized_avg_dist * average_distance_factor)
        metrics["average distance to final point"] = contrib
        error += contrib

    # 4. Convergence speed.
    if convergence_factor > 0.0:
        actual_convergence_tol = convergence_tol * search_space_diag
        dists = torch.min(
            torch.norm(steps.T[:, None, :] - global_min_pos[None, :, :], dim=2), dim=1
        ).values
        hits = torch.nonzero(dists < actual_convergence_tol, as_tuple=True)[0]
        first_hit = hits[0].item() if len(hits) > 0 else num_iters
        normalized_hit_time = first_hit / num_iters
        contrib = error_scaler(normalized_hit_time * convergence_factor)
        metrics["convergence"] = contrib
        error += contrib

    # 5. Oscillation penalty.
    if oscillation_factor > 0.0:
        step_vecs = steps[:, 1:] - steps[:, :-1]
        step_norms = torch.norm(step_vecs, dim=0, keepdim=True)
        unit_vecs = step_vecs / (step_norms + 1e-12)
        sharp_turns = torch.clamp(
            -(unit_vecs[:, 1:] * unit_vecs[:, :-1]).sum(0), min=0.0
        )
        normalized_mean_step_size = step_norms.mean().item() / search_space_diag
        penalty = sharp_turns.mean().item() * normalized_mean_step_size
        contrib = error_scaler(penalty * oscillation_factor)
        metrics["oscillation"] = contrib
        error += contrib

    # 6. Large step ("lucky jump") penalty.
    if lucky_jump_factor > 0.0:
        largest_step = torch.norm(steps[:, 1:] - steps[:, :-1], dim=0).max().item()
        observed_span = (steps.max(1).values - steps.min(1).values).max().item()
        rel_step = largest_step / max(search_space_diag, observed_span)

        if rel_step > lucky_jump_threshold:
            delta = (rel_step - lucky_jump_threshold) / lucky_jump_threshold
            contrib = error_scaler((delta**2) * lucky_jump_factor)
            metrics["lucky jump"] = contrib
            error += contrib

    # 7. Insufficient movement penalty
    if min_movement_factor > 0.0:
        pts = steps.T.numpy()
        area = ConvexHull(pts).volume

        max_area = (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0])
        normalized_area = area / max_area

        if normalized_area < min_movement_threshold:
            delta = min_movement_threshold - normalized_area
            contrib = error_scaler(delta * delta * min_movement_factor)
            metrics["min movement (area)"] = contrib
            error += contrib

    # 8. Final position too close to start penalty.
    if final_proximity_factor > 0.0:
        final_disp = torch.norm(final_pos - start_pos).item()
        if final_disp < final_proximity_threshold * max_side:
            delta = (final_proximity_threshold * max_side - final_disp) / max_side
            exp_penalty = (torch.exp(torch.tensor(delta * 10.0)) - 1.0).item()
            contrib = error_scaler(exp_penalty * final_proximity_factor)
            metrics["final proximity"] = contrib
            error += contrib

    if debug:
        print("[objective] contributions:", metrics, "=> total:", error)

    return float("inf") if math.isnan(error) else error, metrics
