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
    average_distance_factor: float = 1.8,
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
        boundary_penalty (bool, optional): If True, penalize positions outside the bounds. Defaults to True.
        average_distance_factor (float, optional): Weight for the average distance penalty. Defaults to 1.8.
        **eval_args: Additional arguments passed to the step execution function.

    Returns:
        float: Total error, combining distance to global minima, average distance along trajectory,
               and optional boundary penalty. Lower values indicate better optimization performance.
    """
    cords = Pos2D(criterion, start_pos)
    optimizer = optimizer_maker(cords, optimizer_conf)
    steps = execute_steps(cords, optimizer, num_iters, **eval_args)
    error: float = 0.0

    if boundary_penalty:
        error += (
            torch.clamp(bounds[0][0] - steps[0], min=0).max()
            + torch.clamp(steps[0] - bounds[0][1], min=0).max()
            + torch.clamp(bounds[1][0] - steps[1], min=0).max()
            + torch.clamp(steps[1] - bounds[1][1], min=0).max()
        ).item() ** 2

    # Distance from global minimum
    error += torch.min(torch.norm(global_min_pos - steps[:, -1], dim=1)).item()

    # Average distance from global minimum
    error += (
        torch.min(
            torch.norm(steps.T[:, None, :] - global_min_pos[None, :, :], dim=2), dim=1
        )
        .values.mean()
        .item()
    ) * average_distance_factor

    return error
