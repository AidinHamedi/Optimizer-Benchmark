import warnings

import torch

from .model import Pos2D

# This warning is common with second-order optimizers that use create_graph=True.
# It's not relevant to the benchmark's correctness, so we suppress it to keep the output clean.
warnings.filterwarnings(
    "ignore", category=UserWarning, module=r"torch\.autograd\.graph"
)


def execute_steps(
    model: Pos2D,
    optimizer: torch.optim.Optimizer,
    num_iters: int,
    use_closure: bool = False,
    use_graph: bool = False,
):
    """Executes optimization steps and records the trajectory."""
    cords = torch.zeros((2, num_iters + 1), dtype=torch.float32)
    cords[:, 0] = model.cords.detach()

    for i in range(1, num_iters + 1):
        if use_closure:
            # A closure is a function that re-evaluates the model and returns the loss.
            # Some optimizers, like LBFGS or those that need to compute the loss multiple
            # times per step, require it.
            def closure():
                optimizer.zero_grad()

                loss = model()
                loss.backward(create_graph=use_graph)

                return loss.detach()

            optimizer.step(closure)

            cords[:, i] = model.cords.detach()
        else:
            loss = model()
            loss.backward(create_graph=use_graph)

            optimizer.step()

            optimizer.zero_grad()

            cords[:, i] = model.cords.detach()

    return cords
