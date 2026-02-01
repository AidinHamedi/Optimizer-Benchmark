import torch
from torch import nn


class HeightZ(torch.Tensor):
    """1D tensor for vertical Z-axis values.

    ~~~

    Args:
        data: The tensor data representing a single Z-axis value.
        *args: Additional positional arguments for torch.Tensor.
        **kwargs: Additional keyword arguments for torch.Tensor.

    Returns:
        A HeightZ instance as a subclass of torch.Tensor.

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
    """2D position tensor (x, y) with attribute access.

    ~~~

    Args:
        data: The tensor data representing (x, y) coordinates.
        *args: Additional positional arguments for torch.Tensor.
        **kwargs: Additional keyword arguments for torch.Tensor.

    Returns:
        A Pos2D instance as a subclass of torch.Tensor.

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

    @property
    def x(self) -> torch.Tensor:
        """Returns the X coordinate.

        Returns:
            torch.Tensor: The value at index 0.
        """
        return self[0]

    @property
    def y(self) -> torch.Tensor:
        """Returns the Y coordinate.

        Returns:
            torch.Tensor: The value at index 1.
        """
        return self[1]


class LearnablePos2D(Pos2D):
    """Subclass of Pos2D for nn.Parameter objects.

    Used to bridge the gap between static coordinate tensors and
    learnable parameters in PyTorch.

    Args:
        param: The neural network parameter to wrap.

    Raises:
        TypeError: If param is not an instance of nn.Parameter.
    """

    def __init__(self, param: nn.Parameter):
        if not isinstance(param, nn.Parameter):
            raise TypeError("LearnablePos2D must wrap nn.Parameter")


class Minimum(Pos2D):
    """Represents a specific global minimum point."""

    pass
