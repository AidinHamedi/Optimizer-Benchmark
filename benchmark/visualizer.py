from typing import Tuple, Union

import matplotlib.pyplot as plt
import torch

from .functions import scale_eval_size
from .utils.surface import compute_surface


def plot_function(
    func,
    func_name: str,
    cords: torch.Tensor,
    output_file: str,
    optimizer_name: str,
    optimizer_params: dict,
    error_rate: float,
    global_minimums: torch.Tensor,
    eval_size: Tuple[Tuple[int, int], Tuple[int, int]],
    res: Union[int, str] = "auto",
):
    """
    Visualizes an optimizer's trajectory over a 2D function surface.

    Args:
        func: The 2D mathematical function to visualize
        func_name: Name of the function (for plot title and caching)
        cords: Tensor containing the optimization trajectory coordinates
        output_file: Path where the plot image will be saved
        optimizer_name: Name of the optimizer used
        optimizer_params: Dictionary of optimizer parameters (shown in plot title)
        global_minimums: Tensor containing the global minimum point(s)
        eval_size: Tuple defining the x and y axis ranges
        res: Resolution of the surface plot (points per axis)
    """
    X, Y, Z = compute_surface(func, func_name, scale_eval_size(eval_size, 1.1), res)

    fig, ax = plt.subplots(figsize=(14, 14))
    cs = ax.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=20, cmap="jet")
    fig.colorbar(cs, ax=ax, label="f(x, y)")

    xs = cords[0].numpy()
    ys = cords[1].numpy()

    ax.plot(xs, ys, color="black", linewidth=2, alpha=0.8, marker="x", label="path")
    ax.scatter(
        xs[0],
        ys[0],
        marker="o",
        s=80,
        color="lime",
        edgecolor="black",
        linewidths=0.8,
        zorder=5,
        label="start",
    )
    ax.scatter(
        xs[-1],
        ys[-1],
        marker="X",
        s=110,
        color="red",
        edgecolor="black",
        linewidths=0.8,
        zorder=6,
        label="final",
    )

    gx = global_minimums[:, 0].numpy()
    gy = global_minimums[:, 1].numpy()

    ax.scatter(
        gx,
        gy,
        marker="*",
        s=140,
        color="gold",
        edgecolor="black",
        linewidths=0.8,
        zorder=7,
        label="global minimum",
    )

    config = ", ".join(
        f"{k}={round(v, 4) if isinstance(v, (int, float)) else v}"
        for k, v in optimizer_params.items()
    )

    iterations = max(0, len(xs) - 1)
    ax.set_title(
        f"{func_name} | {optimizer_name}\n"
        f"Iterations: {iterations}, Error: {error_rate:.4f}\n"
        f"Config: {config}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")

    plt.savefig(output_file, bbox_inches="tight", dpi=120)
    plt.close()
