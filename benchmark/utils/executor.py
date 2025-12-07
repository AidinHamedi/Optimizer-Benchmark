import warnings
from typing import Any, Callable, Dict

import torch

from .model import Pos2D

warnings.filterwarnings("ignore", category=UserWarning)


def execute_steps(
    model: Pos2D,
    optimizer: torch.optim.Optimizer,
    num_iters: int,
    use_closure: bool = False,
    use_graph: bool = False,
):
    """Execute optimization steps and record the trajectory.

    Args:
        model: The 2D position model to optimize.
        optimizer: The optimizer instance to use.
        num_iters: Number of optimization iterations.
        use_closure: Use a closure function for the optimizer step.
        use_graph: Create graph during backward pass (for second-order optimizers).

    Returns:
        Tensor of shape [2, num_iters + 1] containing the trajectory coordinates.
    """
    cords = torch.zeros((2, num_iters + 1), dtype=torch.float32)
    cords[:, 0] = model.cords.detach()

    def closure():
        optimizer.zero_grad()

        loss = model()
        loss.backward(create_graph=use_graph)

        return loss

    for i in range(1, num_iters + 1):
        if use_closure:
            optimizer.step(closure)

            cords[:, i] = model.cords.detach()
        else:
            optimizer.zero_grad()

            loss = model()
            loss.backward(create_graph=use_graph)

            optimizer.step()

            cords[:, i] = model.cords.detach()

    return cords


def optimize(
    criterion: Callable[[torch.Tensor], torch.Tensor],
    optimizer_maker: Callable[[Pos2D, Dict, int], Any],
    optimizer_conf: Dict[str, Any],
    start_pos: torch.Tensor,
    num_iters: int,
    eval_args: Dict[str, Any],
):
    """Run optimization and return the trajectory.

    Args:
        criterion: The objective function to minimize.
        optimizer_maker: Factory function that creates an optimizer instance.
        optimizer_conf: Configuration dictionary for the optimizer.
        start_pos: Starting position as a tensor [x, y].
        num_iters: Number of optimization iterations.
        eval_args: Additional arguments (use_closure, use_graph) for execution.

    Returns:
        Tensor of shape [2, num_iters + 1] containing the trajectory coordinates.

    Raises:
        ValueError: If the optimizer produces NaN or Inf values.
    """
    cords = Pos2D(criterion, start_pos)
    optimizer = optimizer_maker(cords, optimizer_conf, num_iters)

    steps = execute_steps(cords, optimizer, num_iters, **eval_args)

    if not torch.isfinite(steps).all():
        raise ValueError("Optimizer generated NaN or Inf values.")

    return steps
