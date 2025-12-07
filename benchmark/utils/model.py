from typing import Callable

import torch
from torch import nn


class Pos2D(nn.Module):
    """A 2D position model for optimizer benchmarking.

    Wraps a 2D coordinate as a learnable parameter and evaluates it
    against a given objective function.
    """

    def __init__(self, func: Callable, start_pos: torch.Tensor) -> None:
        """Initialize the 2D position model.

        Args:
            func: Objective function that takes a 2D tensor and returns a scalar.
            start_pos: Initial coordinates as a tensor [x, y].
        """
        super().__init__()
        self.func = func
        self.cords: torch.Tensor = nn.Parameter(
            start_pos.to(dtype=torch.float32, non_blocking=True, copy=True),
            requires_grad=True,
        )

    def forward(self) -> torch.Tensor:
        """Evaluate the objective function at the current position.

        Returns:
            Scalar tensor with the function value.
        """
        return self.func(self.cords)
