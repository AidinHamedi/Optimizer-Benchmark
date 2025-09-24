from functools import wraps
from typing import Callable

import torch


def normalize(
    min_val: float, max_val: float, out_min: float = 0.0, out_max: float = 2.0
):
    """
    Returns a decorator that normalizes the output of a function from [min_val, max_val]
    to [out_min, out_max] using PyTorch operations.

    Args:
        min_val (float): Minimum expected function output.
        max_val (float): Maximum expected function output.
        out_min (float): Target lower bound (default: 0.0).
        out_max (float): Target upper bound (default: 2.0).
    """
    out_range = out_max - out_min

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs) -> torch.Tensor:
            raw: torch.Tensor = func(*args, **kwargs)

            min_t = torch.as_tensor(min_val, dtype=raw.dtype, device=raw.device)
            max_t = torch.as_tensor(max_val, dtype=raw.dtype, device=raw.device)
            out_min_t = torch.as_tensor(out_min, dtype=raw.dtype, device=raw.device)
            out_range_t = torch.as_tensor(out_range, dtype=raw.dtype, device=raw.device)

            norm = (raw - min_t) / (max_t - min_t)
            scaled = norm * out_range_t + out_min_t
            return scaled

        return wrapper

    return decorator
