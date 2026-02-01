from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

from .types import Minimum, Pos2D


@dataclass(frozen=True)
class ProblemBounds:
    """Defines the spatial boundaries for an optimization problem.

    Attributes:
        x_range: A tuple containing (min, max) for the X axis.
        y_range: A tuple containing (min, max) for the Y axis.
    """

    x_range: tuple[float, float]
    y_range: tuple[float, float]

    def __post_init__(self):
        """Validates that min values are less than max values."""
        if self.x_range[0] >= self.x_range[1] or self.y_range[0] >= self.y_range[1]:
            raise ValueError("Range min must be less than max.")

    @property
    def x_min(self) -> float:
        """float: The minimum boundary of the X axis."""
        return self.x_range[0]

    @property
    def x_max(self) -> float:
        """float: The maximum boundary of the X axis."""
        return self.x_range[1]

    @property
    def y_min(self) -> float:
        """float: The minimum boundary of the Y axis."""
        return self.y_range[0]

    @property
    def y_max(self) -> float:
        """float: The maximum boundary of the Y axis."""
        return self.y_range[1]

    def in_bounds(self, pos: Pos2D) -> bool:
        """Checks if a given 2D position is within the defined bounds.

        Args:
            pos: The 2D position tensor to check.

        Returns:
            bool: True if position is within x and y ranges, False otherwise.
        """
        x_ok = self.x_min <= pos.x.item() <= self.x_max
        y_ok = self.y_min <= pos.y.item() <= self.y_max
        return x_ok and y_ok

    def scale(self, ratio: float) -> "ProblemBounds":
        """Scales boundaries from the center by ratio."""

        def _scale(r: tuple[float, float]) -> tuple[float, float]:
            center = (r[0] + r[1]) / 2

            half_width = abs(r[1] - r[0]) * ratio / 2
            return (center - half_width, center + half_width)

        return ProblemBounds(_scale(self.x_range), _scale(self.y_range))


@dataclass(frozen=True)
class GlobalMinima:
    """Collection of global minima for an optimization problem.

    Attributes:
        points: A list of Minimum (Pos2D) objects.
    """

    points: list[Minimum] = field(default_factory=list)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "GlobalMinima":
        """Creates a collection from a tensor of shape (N, 2)."""
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return cls(points=[Minimum(p) for p in tensor])

    def __iter__(self) -> Iterable[Minimum]:
        return iter(self.points)

    def __len__(self) -> int:
        return len(self.points)

    def as_tensor(self) -> torch.Tensor:
        """Returns all minima as a single stacked tensor."""
        return torch.stack([p for p in self.points])


@dataclass
class FuncConstants:
    """Function hyperparameters (e.g., coefficients)."""

    pass


@dataclass
class OptimizationProblemConf:
    """Configuration settings for an optimization problem instance.

    Attributes:
        func_name: The display name of the function.
        start_pos: The initial coordinates for the optimizer.
        bounds: The spatial limits for the search space.
        global_minima: Collection of known global minimum coordinates.
        constants: Function hyperparameters (e.g., coefficients).
    """

    func_name: str
    start_pos: Pos2D
    bounds: ProblemBounds
    global_minima: GlobalMinima
    constants: FuncConstants
