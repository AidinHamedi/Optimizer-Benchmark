import torch
from torch import nn

from .schema import OptimizationProblemConf
from .types import HeightZ, LearnablePos2D


class OptimizationProblem2D(nn.Module):
    """Base class for 2D optimization problems.

    Attributes:
        conf (OptimizationProblemConf): The configuration object for this problem.
        pos (LearnablePos2D): A learnable 2D position wrapper.
    """

    def __init__(self, conf: OptimizationProblemConf) -> None:
        """Initializes the problem with a starting position and config.

        Args:
            conf: The optimization problem configuration.
        """
        super().__init__()
        self.conf = conf

        self._raw_pos = nn.Parameter(
            conf.start_pos.to(dtype=torch.float32, copy=True),
            requires_grad=True,
        )

        self.pos = LearnablePos2D(self._raw_pos)

        self.func_init()

    def func(self, pos: LearnablePos2D) -> HeightZ:
        """The objective function to be minimized.

        Args:
            pos: The current 2D position.

        Returns:
            HeightZ: The resulting value at the given position.

        Raises:
            NotImplementedError: If not overridden by a subclass.
        """
        raise NotImplementedError("Implement objective logic in subclass.")

    def func_init(self) -> None:
        """For initialization of the objective function."""
        pass

    def forward(self) -> HeightZ:
        """Standard PyTorch forward pass returning evaluation at current position."""
        return self.func(self.pos)
