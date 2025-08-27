import math
from pathlib import Path
from typing import Callable, Tuple, Union

import torch


def compute_surface(
    func: Callable,
    func_name: str,
    eval_size: Tuple[Tuple[int, int], Tuple[int, int]],
    res: Union[int, str] = "auto",
    cache: bool = True,
    cache_dir: str = "./cache",
) -> torch.Tensor:
    """
    Computes the surface of a 2D function and returns it as a tensor.

    Args:
        func (Callable): The function to evaluate.
        eval_size (Tuple[Tuple[int, int], Tuple[int, int]]): The evaluation range ((x_min, x_max), (y_min, y_max)).
        res (Union[int, str], optional): The resolution for the grid. Defaults to "auto".
        cache (bool, optional): Whether to cache the resulting tensor. Defaults to True.
        cache_dir (str, optional): The directory to store cached tensors. Defaults to "./cache".

    Returns:
        torch.Tensor: A tensor of shape (3, res, res) containing X, Y, and Z coordinates.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"{func_name}.pt"

    if cache and cache_file.exists():
        return torch.load(cache_file)

    if res == "auto":
        num_points = int(math.sqrt(eval_size[0][1]) * 200)
    else:
        num_points = int(res)

    x_bounds, y_bounds = eval_size
    x = torch.linspace(x_bounds[0] * 1.2, x_bounds[1] * 1.2, num_points)
    y = torch.linspace(y_bounds[0] * 1.2, y_bounds[1] * 1.2, num_points)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)

    with torch.no_grad():
        try:
            Z = func(grid_points).reshape(num_points, num_points)
        except Exception:
            Z_vals = [func(p) for p in grid_points]
            Z = torch.stack(Z_vals).reshape(num_points, num_points)

    surface_tensor = torch.stack([X, Y, Z], dim=0)

    if cache:
        torch.save(surface_tensor, cache_file)

    return surface_tensor
