from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

type Pair[T] = tuple[T, T]


class HeightZ(torch.Tensor):
    """1D tensor for vertical Z-axis values.

    Args:
        data: The tensor data representing a single Z-axis value.

    Raises:
        TypeError: If data is not a torch.Tensor.
        ValueError: If data does not contain exactly one element.
    """

    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        if not isinstance(data, torch.Tensor):
            raise TypeError("HeightZ must wrap a torch.Tensor")
        if data.numel() != 1:
            raise ValueError("HeightZ must contain one value.")
        return torch.as_tensor(data, dtype=torch.float32).as_subclass(cls)


class Pos2D(torch.Tensor):
    """2D position tensor (x, y).

    Args:
        data: The tensor data representing (x, y) coordinates.

    Raises:
        TypeError: If data is not a torch.Tensor.
        ValueError: If data shape is not (2,).
    """

    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        if not isinstance(data, torch.Tensor):
            raise TypeError("Pos2D must wrap a torch.Tensor")
        if data.shape != (2,):
            raise ValueError(f"Pos2D must be shape (2,), got {data.shape}")
        return torch.as_tensor(data, dtype=torch.float32).as_subclass(cls)

    @classmethod
    def from_pair(cls, data: Pair[float]) -> "Pos2D":
        """Creates a Pos2D instance from a tuple of coordinates."""
        if not isinstance(data, tuple):
            raise TypeError(f"Expected tuple, got {type(data).__name__}")

        if len(data) != 2:
            raise ValueError(f"Tuple must have exactly 2 elements, got {len(data)}")

        if not all(isinstance(val, (int, float)) for val in data):
            raise TypeError("All tuple elements must be numeric (int or float)")

        tensor = torch.tensor([*data], dtype=torch.float32)
        return cls(tensor)

    @property
    def x(self) -> torch.Tensor:
        """Returns the X coordinate value at index 0."""
        return self[0]

    @property
    def y(self) -> torch.Tensor:
        """Returns the Y coordinate value at index 1."""
        return self[1]


class LearnablePos2D(Pos2D):
    """Subclass of Pos2D used to wrap learnable nn.Parameter objects."""

    def __init__(self, param: nn.Parameter):
        if not isinstance(param, nn.Parameter):
            raise TypeError("LearnablePos2D must wrap nn.Parameter")

    def to_pos2d(self) -> Pos2D:
        """Returns a detached, non-gradient version of the position as a Pos2D instance."""
        return self.detach().as_subclass(Pos2D)


class Trajectory(torch.Tensor):
    """A fixed-capacity buffer for tracking 2D positions (N, 2).

    Attributes:
        _count: The number of steps currently recorded.
        capacity: The total allocated size of the buffer.
    """

    _count: int
    capacity: int

    @staticmethod
    def __new__(cls, capacity: int, *args, **kwargs):
        """Initializes the trajectory buffer.

        Args:
            capacity: Maximum number of 2D coordinates to store.

        Returns:
            Trajectory: A float32 tensor instance of shape (capacity, 2).
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be > 0, got {capacity}")

        instance = torch.zeros((capacity, 2), dtype=torch.float32).as_subclass(cls)
        instance.capacity = capacity
        instance._count = 0
        return instance

    def append(self, pos: Pos2D) -> None:
        """Appends a position in-place using fast data assignment.

        Args:
            pos: The 2D position to record.

        Raises:
            ValueError: If capacity is exceeded.
        """
        if self._count >= self.capacity:
            raise ValueError(f"Trajectory capacity ({self.capacity}) reached.")

        self.data[self._count] = pos.data
        self._count += 1

    def get_step(self, index: int) -> Pos2D:
        """Returns the position at a specific index.

        Raises:
            IndexError: If the index has not been populated yet.
        """
        if not 0 <= index < self._count:
            raise IndexError(
                f"Index {index} out of recorded range (0-{self._count - 1})."
            )

        return Pos2D(self[index])

    def reset(self) -> None:
        """Resets the record count without reallocating memory."""
        self._count = 0

    @property
    def path(self) -> torch.Tensor:
        """torch.Tensor: Returns the active slice of the trajectory (steps, 2)."""
        return self[: self._count]

    @property
    def steps(self) -> int:
        """int: The number of positions currently in the trajectory."""
        return self._count

    @property
    def is_full(self) -> bool:
        """bool: True if no more steps can be added."""
        return self._count == self.capacity


@dataclass(frozen=True)
class ProblemBounds:
    """Defines the spatial boundaries for an optimization problem.

    Attributes:
        x_range: A tuple containing (min, max) for the X axis.
        y_range: A tuple containing (min, max) for the Y axis.
    """

    x_range: Pair[float]
    y_range: Pair[float]

    def __post_init__(self):
        """Validates that min values are less than max values."""
        if self.x_range[0] >= self.x_range[1] or self.y_range[0] >= self.y_range[1]:
            raise ValueError("Range min must be less than max.")

    @classmethod
    def from_pair(cls, bounds_tuple: Pair[Pair[float]]) -> "ProblemBounds":
        """Creates ProblemBounds from a nested tuple structure."""
        if not isinstance(bounds_tuple, tuple):
            raise TypeError(f"Expected tuple, got {type(bounds_tuple).__name__}")

        if len(bounds_tuple) != 2:
            raise ValueError(
                f"Expected tuple with 2 elements (x_range, y_range), got {len(bounds_tuple)}"
            )

        x_range, y_range = bounds_tuple

        # Validate inner tuples
        for name, rng in [("x_range", x_range), ("y_range", y_range)]:
            if not isinstance(rng, tuple):
                raise TypeError(f"{name} must be a tuple, got {type(rng).__name__}")
            if len(rng) != 2:
                raise ValueError(f"{name} must have exactly 2 elements, got {len(rng)}")

        return cls(x_range, y_range)

    @property
    def x_min(self) -> float:
        """The minimum boundary of the X axis."""
        return self.x_range[0]

    @property
    def x_max(self) -> float:
        """The maximum boundary of the X axis."""
        return self.x_range[1]

    @property
    def y_min(self) -> float:
        """The minimum boundary of the Y axis."""
        return self.y_range[0]

    @property
    def y_max(self) -> float:
        """The maximum boundary of the Y axis."""
        return self.y_range[1]

    def in_bounds(self, pos: Pos2D) -> bool:
        """Checks if a given 2D position is within the defined ranges."""
        x_ok = self.x_min <= pos.x.item() <= self.x_max
        y_ok = self.y_min <= pos.y.item() <= self.y_max
        return x_ok and y_ok

    def scale(self, ratio: float) -> "ProblemBounds":
        """Scales boundaries from the center by a given ratio."""

        def _scale(r: tuple[float, float]) -> tuple[float, float]:
            center = (r[0] + r[1]) / 2

            half_width = abs(r[1] - r[0]) * ratio / 2
            return (center - half_width, center + half_width)

        return ProblemBounds(_scale(self.x_range), _scale(self.y_range))


@dataclass(frozen=True)
class GlobalMinima:
    """Collection of global minima points for an optimization problem."""

    points: list[Pos2D] = field(default_factory=list)

    @classmethod
    def from_list(cls, raw_list: list) -> "GlobalMinima":
        if not raw_list:
            return cls(points=[])

        points = [
            Pos2D.from_pair(p) if not isinstance(p, Pos2D) else p for p in raw_list
        ]
        return cls(points=points)

    def as_tensor(self) -> torch.Tensor:
        """Returns all minima as a single stacked tensor of shape (N, 2)."""
        return torch.stack([p for p in self.points])

    @property
    def is_unknown(self) -> bool:
        """Returns True if no global minima are defined."""
        return len(self.points) == 0

    def __iter__(self) -> Iterable[Pos2D]:
        return iter(self.points)

    def __len__(self) -> int:
        return len(self.points)


@dataclass(frozen=True)
class FuncConstants:
    """Dynamic container for function-specific hyperparameters."""

    def __init__(self, data: dict[str, Any]):
        if not isinstance(data, dict):
            raise TypeError(f"FuncConstants expects a dict, got {type(data).__name__}")

        # Since it's frozen, we must use object.__setattr__
        for key, value in data.items():
            object.__setattr__(self, key, value)

    @classmethod
    def from_dict(cls, data: Any) -> "FuncConstants":
        return cls(data if data is not None else {})


@dataclass(frozen=True)
class NormParams:
    """Normalization parameters containing scalar and tensor representations.

    Attributes:
        max: The maximum normalization value.
        min: The minimum normalization value.
    """

    @dataclass(frozen=True)
    class _Value:
        scalar: float
        tensor: torch.Tensor

    max: _Value
    min: _Value

    @classmethod
    def from_pair(cls, bounds: Pair[float]) -> "NormParams":
        """Creates a normalization parameters object from a tuple of bounds."""
        min_val, max_val = bounds

        min_tensor = torch.tensor(min_val, dtype=torch.float32)
        max_tensor = torch.tensor(max_val, dtype=torch.float32)

        return cls(
            max=cls._Value(max_val, max_tensor),
            min=cls._Value(min_val, min_tensor),
        )
