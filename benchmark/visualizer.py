from typing import Callable, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

from .functions import scale_eval_size
from .utils.surface import compute_surface


def plot_function(
    func: Callable,
    func_name: str,
    cords: torch.Tensor,
    output_file: str,
    optimizer_name: str,
    optimizer_params: dict,
    metrics: dict,
    error_rate: float,
    global_minimums: torch.Tensor,
    eval_size: Tuple[Tuple[int, int], Tuple[int, int]],
    res: Union[int, str] = "auto",
    debug: bool = False,
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
        metrics: Dictionary of penalty breakdowns to display in a legend
        error_rate: The calculated total error/penalty score
        global_minimums: Tensor containing the global minimum point(s)
        eval_size: Tuple defining the x and y axis ranges
        res: Resolution of the surface plot (points per axis)
        debug: Debug mode flag
    """
    X, Y, Z = compute_surface(
        func, func_name, scale_eval_size(eval_size, 1.1), res, debug=debug
    )

    fig, ax = plt.subplots(figsize=(14, 14))
    cs = ax.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=25, cmap="jet")
    fig.colorbar(cs, ax=ax, label="f(x, y)")

    xs = cords[0].numpy()
    ys = cords[1].numpy()

    # Trajectory plotting
    ax.plot(
        xs,
        ys,
        color="black",
        linewidth=2,
        alpha=1,
        marker="o",
        label="Path",
        markersize=5,
        markerfacecolor="white",
    )
    ax.scatter(
        xs[0],
        ys[0],
        marker="o",
        s=80,
        color="lime",
        edgecolor="black",
        linewidths=0.8,
        zorder=5,
        label="Start",
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
        label="Final",
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
        label="Global Min",
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

    # 1. Main Legend (Path items)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    main_legend = ax.legend(by_label.values(), by_label.keys(), loc="upper left")
    ax.add_artist(main_legend)

    # 2. Metrics Legend (Penalties)
    # Filter out zero values and sort by magnitude (descending)
    active_metrics = [
        (k, v) for k, v in metrics.items() if isinstance(v, (int, float)) and v > 0
    ]
    active_metrics.sort(key=lambda x: x[1], reverse=True)

    if active_metrics:
        metric_handles = []
        for k, v in active_metrics:
            # Create invisible patches to act as text holders
            clean_name = k.replace("_", " ").title()
            patch = mpatches.Patch(color="none", label=f"{clean_name}: {v:.4f}")
            metric_handles.append(patch)

        ax.legend(
            handles=metric_handles,
            loc="lower right",
            title="Penalty Breakdown",
        )

    plt.savefig(output_file, bbox_inches="tight", dpi=120)
    plt.close()
