from typing import Callable, Tuple

import torch
from torch import nn


class Pos2D(nn.Module):
    """
    Simple 2D optimization model.
    """

    def __init__(self, func: Callable, start_pos: torch.Tensor) -> None:
        """
        Args:
            func: Mathematical function to optimize.
            start_pos: Starting point for optimization (x, y).
        """
        super().__init__()
        self.func = func
        self.cords: torch.Tensor = nn.Parameter(
            start_pos.to(dtype=torch.float32, non_blocking=True, copy=True),
            requires_grad=True,
        )

    def forward(self) -> torch.Tensor:
        return self.func(self.cords)
