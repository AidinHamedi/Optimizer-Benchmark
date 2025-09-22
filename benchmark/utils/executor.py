import warnings

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
