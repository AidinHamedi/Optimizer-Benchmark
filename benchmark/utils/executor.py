import warnings

import torch
from torch import nn

from .model import Pos2D

warnings.filterwarnings(
    "ignore", category=UserWarning, module=r"torch\.autograd\.graph"
)


def execute_steps(
    model: Pos2D,
    optimizer: torch.optim.Optimizer,
    num_iters: int,
    use_closure: bool = False,
    use_graph: bool = False,
    grad_clip: bool = True,
    grad_clip_value: float = 1.0,
):
    """Executes optimization steps and records the trajectory.

    Args:
        model (Pos2D): The model to be optimized.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        num_iters (int): The number of optimization iterations.
        use_closure (bool, optional): Whether to use a closure for the optimizer. Defaults to False.
        use_graph (bool, optional): Whether to create a graph of the backward pass. Defaults to False.
        grad_clip (bool, optional): Whether to clip gradients. Defaults to True.
        grad_clip_value (float, optional): The value to clip gradients to. Defaults to 1.0.

    Returns:
        torch.Tensor: A tensor of shape (2, num_iters + 1) containing the trajectory of the model's coordinates.
    """
    cords = torch.zeros((2, num_iters + 1), dtype=torch.float32)
    cords[:, 0] = model.cords.detach()

    for i in range(1, num_iters + 1):
        if use_closure:

            def closure():
                optimizer.zero_grad()

                loss = model()
                loss.backward(create_graph=use_graph)

                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

                return loss.detach()

            optimizer.step(closure)

            cords[:, i] = model.cords.detach()
        else:
            loss = model()
            loss.backward(create_graph=use_graph)

            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

            optimizer.step()

            optimizer.zero_grad()

            cords[:, i] = model.cords.detach()

    return cords
