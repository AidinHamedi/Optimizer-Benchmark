import math
from typing import Callable, Dict, Tuple

import torch

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
    average_distance_factor: float = 0.4,
    convergence_factor: float = 0.2,
    convergence_tol: float = 0.1,
    oscillation_factor: float = 1.0,
    lucky_jump_factor: float = 1.0,
    lucky_jump_threshold: float = 0.18,
    final_distance_factor: float = 1.5,
    final_value_factor: float = 0.8,
    min_movement_factor: float = 0.6,
    min_movement_threshold: float = 0.5,
    final_proximity_factor: float = 8.0,
    final_proximity_threshold: float = 0.1,
    debug: bool = False,
    **eval_args,
) -> float:
    """
    Evaluates an optimization trajectory and returns a scalar error.

    The error combines several criteria:
      - Boundary violations (if outside the search space).
      - Distance of the final position from known global minima.
      - Average wandering of the trajectory relative to its final position.
      - Function value (z-value) at the final position.
      - Convergence speed (how quickly it reaches tolerance).
      - Oscillation (sharp turns during trajectory).
      - Large "lucky jump" steps that skip across the search space.
      - Insufficient movement from the starting position (max displacement under threshold).
      - Final position too close to starting position (under threshold).

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
        convergence_tol (float, optional): Distance threshold to consider "converged".
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
        float: Total error (lower is better).
    """
    cords = Pos2D(criterion, start_pos)
    optimizer = optimizer_maker(cords, optimizer_conf, num_iters)

    try:
        steps = execute_steps(cords, optimizer, num_iters, **eval_args)
    except Exception as e:
        # If the optimizer fails (e.g., due to invalid hyperparameters), return infinity
        # to ensure Optuna prunes this trial.
        if debug:
            print(f"Error during optimization: {e}")

        return float("inf")

    final_pos = steps[:, -1]

    error: float = 0.0
    debug_info: Dict[str, float] = {}

    # 0. Function value at final position.
    # This is a primary measure of success: how low did the optimizer get?
    # We clamp at 0 because negative values are good and shouldn't be penalized.
    final_value = max(criterion(final_pos).item(), 0)
    contrib = final_value * final_value_factor
    error += contrib
    if debug:
        debug_info["final_value"] = contrib

    # 1. Boundary penalty: Penalize any steps taken outside the defined search space.
    if boundary_penalty:
        violation = (
            torch.clamp(bounds[0][0] - steps[0], min=0).max()
            + torch.clamp(steps[0] - bounds[0][1], min=0).max()
            + torch.clamp(bounds[1][0] - steps[1], min=0).max()
            + torch.clamp(steps[1] - bounds[1][1], min=0).max()
        )
        # Squaring the violation heavily penalizes larger deviations.
        contrib = violation.item() ** 2
        error += contrib
        if debug:
            debug_info["boundary"] = contrib

    # 2. Final distance to the nearest known global minimum.
    # This directly measures how close the optimizer got to the target.
    final_dist = torch.min(torch.norm(global_min_pos - final_pos, dim=1)).item()
    contrib = final_dist * final_distance_factor
    error += contrib
    if debug:
        debug_info["final_distance"] = contrib

    # 3. Average trajectory distance from the final point.
    # This penalizes "wandering" behavior. A good optimizer should move decisively
    # towards the minimum, not explore randomly around its final destination.
    avg_dist = torch.norm(steps.T - final_pos[None, :], dim=1).mean().item()
    contrib = avg_dist * average_distance_factor
    error += contrib
    if debug:
        debug_info["average_distance"] = contrib

    # 4. Convergence speed (optional): How many steps did it take to get close?
    # We reward optimizers that find the minimum area quickly.
    if convergence_factor > 0.0:
        dists = torch.min(
            torch.norm(steps.T[:, None, :] - global_min_pos[None, :, :], dim=2), dim=1
        ).values
        hits = torch.nonzero(dists < convergence_tol, as_tuple=True)[0]
        # If it never converges, penalize it as if it took the max number of iterations.
        first_hit = hits[0].item() if len(hits) > 0 else num_iters
        normalized_hit = first_hit / num_iters
        contrib = normalized_hit * convergence_factor
        error += contrib
        if debug:
            debug_info["convergence"] = contrib

    # 5. Oscillation penalty (optional): Penalize sharp, unproductive turns.
    # A high dot product between consecutive step vectors indicates a sharp turn.
    # We scale this by the average step size to give more weight to oscillations with larger movements.
    if oscillation_factor > 0.0:
        step_vecs = steps[:, 1:] - steps[:, :-1]
        unit_vecs = step_vecs / (torch.norm(step_vecs, dim=0, keepdim=True) + 1e-8)
        # Dot product of consecutive unit vectors. -1 is a 180-degree turn.
        # We clamp at 0 since we only care about turns > 90 degrees.
        sharp_turns = torch.clamp(
            -(unit_vecs[:, 1:] * unit_vecs[:, :-1]).sum(0), min=0.0
        )
        penalty = sharp_turns.mean().item() * torch.norm(step_vecs, dim=0).mean().item()
        contrib = penalty * oscillation_factor
        error += contrib
        if debug:
            debug_info["oscillation"] = contrib

    # 6. Large step ("lucky jump") penalty (optional).
    # This penalizes optimizers that succeed by taking a massive leap across the search
    # space, which is often unstable and not a desirable trait.
    if lucky_jump_factor > 0.0:
        largest_step = torch.norm(steps[:, 1:] - steps[:, :-1], dim=0).max().item()
        ranges = torch.tensor(
            [bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]],
            dtype=torch.float32,
        )
        max_side = ranges.max().item()
        diag = torch.norm(ranges).item()
        observed_span = (steps.max(1).values - steps.min(1).values).max().item()

        eps = 1e-12
        # Calculate step size relative to different measures of the space size.
        rel_step = max(
            largest_step / (max_side + eps),
            largest_step / (diag + eps),
            largest_step / (observed_span + eps),
        )

        if rel_step > lucky_jump_threshold:
            delta = (rel_step - lucky_jump_threshold) / lucky_jump_threshold
            contrib = (delta**2) * lucky_jump_factor
            error += contrib
            if debug:
                debug_info["lucky_jump"] = contrib

    # 7. Insufficient movement penalty (optional).
    # Penalize optimizers that "get stuck" and barely move from their start position.
    if min_movement_factor > 0.0:
        max_displacement = torch.norm(steps.T - start_pos[None, :], dim=1).max().item()
        ranges = torch.tensor(
            [bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]],
            dtype=torch.float32,
        )
        max_side = ranges.max().item()
        if max_displacement < min_movement_threshold * max_side:
            delta = (min_movement_threshold * max_side - max_displacement) / (
                max_side + 1e-12
            )
            contrib = (delta**2) * min_movement_factor
            error += contrib
            if debug:
                debug_info["min_movement"] = contrib

    # 8. Final position too close to start penalty (optional).
    # This is similar to #7 but specifically targets the final point. It prevents
    # an optimizer from being rewarded for doing nothing if the start point is already good.
    if final_proximity_factor > 0.0:
        final_disp = torch.norm(final_pos - start_pos).item()
        ranges = torch.tensor(
            [bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]],
            dtype=torch.float32,
        )
        max_side = ranges.max().item()
        if final_disp < final_proximity_threshold * max_side:
            delta = (final_proximity_threshold * max_side - final_disp) / (
                max_side + 1e-12
            )
            # Use an exponential penalty to strongly discourage staying very close to the start.
            exp_penalty = (torch.exp(torch.tensor(delta * 10.0)) - 1.0).item()
            contrib = exp_penalty * final_proximity_factor
            error += contrib
            if debug:
                debug_info["final_proximity"] = contrib

    if debug:
        print("[objective] contributions:", debug_info, "=> total:", error)

    # Return infinity if the error is NaN, which can happen with unstable optimizers.
    return float("inf") if math.isnan(error) else error
