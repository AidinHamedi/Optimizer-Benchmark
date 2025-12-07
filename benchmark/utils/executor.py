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
    """Executes optimization steps and records the trajectory.

    Args:
        model (Pos2D): The model to be optimized.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        num_iters (int): The number of optimization iterations.
        use_closure (bool, optional): Whether to use a closure for the optimizer. Defaults to False.
        use_graph (bool, optional): Whether to create a graph of the backward pass. Defaults to False.

    Returns:
        torch.Tensor: A tensor of shape (2, num_iters + 1) containing the trajectory of the model's coordinates.
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
    """Optimize a model using a given optimizer.

    Args:
        criterion (Callable[[torch.Tensor], torch.Tensor]): The loss function to optimize.
        optimizer_maker (Callable[[Pos2D, Dict, int], Any]): A function that creates an optimizer.
        optimizer_conf (Dict[str, Any]): Configuration for the optimizer.
        start_pos (torch.Tensor): The starting position of the model.
        num_iters (int): The number of iterations to optimize for.
        eval_args (Dict[str, Any]): Additional arguments for the evaluation function.

    Returns:
        torch.Tensor: A tensor of shape (2, num_iters + 1) containing the trajectory of the model's coordinates.
    """
    cords = Pos2D(criterion, start_pos)
    optimizer = optimizer_maker(cords, optimizer_conf, num_iters)

    # Optimize
    steps = execute_steps(cords, optimizer, num_iters, **eval_args)

    # Check for NaN/Inf which implies optimizer explosion
    if not torch.isfinite(steps).all():
        raise ValueError("Optimizer generated NaN or Inf values.")

    return steps
