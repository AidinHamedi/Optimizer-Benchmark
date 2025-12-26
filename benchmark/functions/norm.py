from functools import wraps
from typing import Callable

import torch


def normalize(
    min_val: float, max_val: float, out_min: float = 0.0, out_max: float = 2.0
):
    """Decorator that normalizes function outputs to a target range.

    Args:
        min_val: Expected minimum output of the wrapped function.
        max_val: Expected maximum output of the wrapped function.
        out_min: Target lower bound (default: 0.0).
        out_max: Target upper bound (default: 2.0).

    Returns:
        Decorator that wraps a function to normalize its output.
    """
    scale_factor = (out_max - out_min) / (max_val - min_val)
    scale_tensor = torch.tensor((out_max - out_min) / (max_val - min_val))
    offset_tensor = torch.tensor((out_min - min_val * scale_factor) + 1e-6)

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs) -> torch.Tensor:
            raw = func(*args, **kwargs)
            return torch.addcmul(offset_tensor, raw, scale_tensor)

        return wrapper

    return decorator
