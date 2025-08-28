from typing import Callable, Tuple

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
    average_distance_factor: float = 0.6,
    convergence_factor: float = 0.05,
    oscillation_factor: float = 1.2,
    lucky_jump_factor: float = 2.6,
    lucky_jump_threshold: float = 0.1,
    convergence_tol: float = 1e-2,
    **eval_args,
) -> float:
    """
    Evaluates an optimization trajectory and returns a scalar error.

    Args:
        criterion (Callable): The function to minimize.
        optimizer_maker (Callable): Factory function that creates an optimizer instance.
        optimizer_conf (dict): Configuration parameters for the optimizer.
        start_pos (torch.Tensor): Starting position of the optimization (shape: [2]).
        global_min_pos (torch.Tensor): Tensor of one or more known global minima (shape: [N, 2]).
        bounds (Tuple[Tuple[int, int], Tuple[int, int]]): Allowed range for x and y coordinates.
        num_iters (int): Number of optimization steps to perform.
        boundary_penalty (bool, optional): Penalize positions outside the bounds.
        average_distance_factor (float, optional): Weight for the average distance penalty.
        convergence_factor (float, optional): Weight for convergence speed penalty. 0 disables it.
        oscillation_factor (float, optional): Weight for oscillation penalty. 0 disables it.
        lucky_jump_factor (float, optional): Weight for penalizing very large steps. 0 disables it.
        lucky_jump_threshold (float, optional): Fraction of problem size considered too large.
        convergence_tol (float, optional): Distance threshold for considering a point "converged".
        **eval_args: Additional arguments passed to the step execution function.

    Returns:
        float: Total error, combining multiple metrics. Lower is better.
    """
    cords = Pos2D(criterion, start_pos)
    optimizer = optimizer_maker(cords, optimizer_conf, num_iters)
    steps = execute_steps(cords, optimizer, num_iters, **eval_args)

    error: float = 0.0

    # 1. Boundary penalty
    if boundary_penalty:
        violation = (
            torch.clamp(bounds[0][0] - steps[0], min=0).max()
            + torch.clamp(steps[0] - bounds[0][1], min=0).max()
            + torch.clamp(bounds[1][0] - steps[1], min=0).max()
            + torch.clamp(steps[1] - bounds[1][1], min=0).max()
        )
        error += violation.item() ** 2

    # 2. Final distance to global minimum
    final_dist = torch.min(torch.norm(global_min_pos - steps[:, -1], dim=1)).item()
    error += final_dist

    # 3. Average trajectory distance
    avg_dist = (
        torch.min(
            torch.norm(steps.T[:, None, :] - global_min_pos[None, :, :], dim=2), dim=1
        )
        .values.mean()
        .item()
    )
    error += avg_dist * average_distance_factor

    # 4. Convergence speed (optional)
    if convergence_factor > 0.0:
        dists = torch.min(
            torch.norm(steps.T[:, None, :] - global_min_pos[None, :, :], dim=2), dim=1
        ).values
        hits = torch.nonzero(dists < convergence_tol, as_tuple=True)[0]
        first_hit = hits[0].item() if len(hits) > 0 else num_iters
        normalized_hit = first_hit / num_iters
        error += normalized_hit * convergence_factor

    # 5. Oscillation penalty for sharp turns (> 90°) (optional)
    if oscillation_factor > 0.0:
        step_vecs = steps[:, 1:] - steps[:, :-1]
        unit_vecs = step_vecs / (torch.norm(step_vecs, dim=0, keepdim=True) + 1e-8)
        sharp_turns = torch.clamp(
            -(unit_vecs[:, 1:] * unit_vecs[:, :-1]).sum(0), min=0.0
        )
        penalty = sharp_turns.mean().item() * torch.norm(step_vecs, dim=0).mean().item()
        error += penalty * oscillation_factor

    # 6. Large step penalty (optional)
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
        rel_step = max(
            largest_step / (max_side + eps),
            largest_step / (diag + eps),
            largest_step / (observed_span + eps),
        )

        if rel_step > lucky_jump_threshold:
            delta = (rel_step - lucky_jump_threshold) / lucky_jump_threshold
            error += (delta**2) * lucky_jump_factor

    return error
