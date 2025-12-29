import math
from pathlib import Path
from typing import Callable, Tuple, Union

import torch


def compute_surface(
    func: Callable,
    func_name: str,
    eval_size: Tuple[Tuple[float, float], Tuple[float, float]],
    res: Union[int, str] = "auto",
    cache: bool = True,
    cache_dir: str = "./cache",
    debug: bool = False,
) -> torch.Tensor:
    """Compute a 2D function surface for visualization.

    Args:
        func: The objective function to evaluate.
        func_name: Name of the function (used for cache file naming).
        eval_size: Evaluation range as ((x_min, x_max), (y_min, y_max)).
        res: Grid resolution (points per axis) or "auto" for automatic scaling.
        cache: Enable caching of computed surfaces.
        cache_dir: Directory for storing cached tensors.
        debug: Enable debug output (disables cache loading).

    Returns:
        Tensor of shape [3, res, res] containing X, Y, and Z coordinates.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"{func_name}.pt"

    if cache and cache_file.exists() and not debug:
        return torch.load(cache_file)

    if res == "auto":
        num_points = int(math.sqrt(eval_size[0][1]) * 500)
    else:
        num_points = int(res)

    x_bounds, y_bounds = eval_size
    x = torch.linspace(x_bounds[0], x_bounds[1], num_points)
    y = torch.linspace(y_bounds[0], y_bounds[1], num_points)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)

    print("[Surface] Computing function values...")
    with torch.no_grad():
        try:
            Z = func(grid_points).reshape(num_points, num_points)
        except Exception:
            Z_vals = [func(p) for p in grid_points]
            Z = torch.stack(Z_vals).reshape(num_points, num_points)

    surface_tensor = torch.stack([X, Y, Z], dim=0)

    if cache:
        torch.save(surface_tensor, cache_file)

    if debug:
        idx_flat = int(torch.argmin(Z).item())
        iy, ix = divmod(idx_flat, num_points)
        print(f"[Surface] Surface bounds: {x_bounds}, {y_bounds}")
        print(
            f"[Surface] Generated surface tensor with shape {tuple(surface_tensor.shape)}"
        )
        print(f"[Surface] Z bounds: ({Z.min().item()}, {Z.max().item()})")
        print(
            f"[Surface] Z min at (x={X[iy, ix].item()}, y={Y[iy, ix].item()}) → {Z[iy, ix].item()}"
        )

    return surface_tensor
