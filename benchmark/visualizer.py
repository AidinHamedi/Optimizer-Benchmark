from pathlib import Path
from typing import Callable, Dict, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import LogLocator, NullFormatter

from .functions import scale_eval_size
from .utils.surface import compute_surface

COLORS = {
    "loss": "#1f77b4",
    "grad": "#d62728",
    "step": "#2ca02c",
    "dist": "#9467bd",
    "path": "#7f7f7f",
    "ratio": "#e377c2",
}

METRIC_COLORS = {
    "val_penalty": "#1f77b4",
    "dist_penalty": "#ff7f0e",
    "speed_penalty": "#2ca02c",
    "eff_penalty": "#d62728",
    "bound_penalty": "#9467bd",
    "terrain_violation": "#8c564b",
    "jump_penalty": "#e377c2",
    "prox_penalty": "#7f7f7f",
    "start_prox_penalty": "#7f7f7f",
}


def _eval_z_on_trajectory(func: Callable, pts: np.ndarray) -> np.ndarray:
    """Evaluate function values along a trajectory."""
    pts_tensor = torch.from_numpy(pts).float()
    try:
        vals = func(pts_tensor).numpy()
        if vals.size != pts.shape[0]:
            raise ValueError
        return vals.ravel()
    except Exception:
        return np.array([func(p).item() for p in pts_tensor])


def _compute_gradient_norms(func: Callable, pts: np.ndarray) -> np.ndarray:
    """Compute gradient norm at each point."""
    grads = []
    for p_np in pts:
        p = torch.from_numpy(p_np).float().requires_grad_(True)
        try:
            func(p).sum().backward()
            grads.append(p.grad.norm().item() if p.grad is not None else 0.0)
        except Exception:
            grads.append(0.0)
    return np.array(grads)


def _compute_efficiency(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute displacement, cumulative path length, and step sizes."""
    displacement = np.linalg.norm(pts - pts[0], axis=1)
    step_sizes = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    step_sizes = np.insert(step_sizes, 0, 0.0)
    path_len = np.cumsum(step_sizes)
    return displacement, path_len, step_sizes


def _style_axis(
    ax: plt.Axes,
    title: str,
    xlabel: str = None,
    ylabel: str = None,
    log_scale: bool = False,
):
    """Apply standard clean styling to an axis."""
    ax.set_title(title, fontsize=10, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.tick_params(labelsize=8)

    if log_scale:
        ax.grid(True, which="major", linestyle="-", alpha=0.3)
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    else:
        ax.grid(True, linestyle="--", alpha=0.3)


def _save_surface_plot(
    out_dir: Path,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    scaled_eval_size: Tuple,
    xs: np.ndarray,
    ys: np.ndarray,
    global_minimums: np.ndarray,
    title_text: str,
    metrics: dict,
    img_format: str,
) -> str:
    """Save the main surface contour and trajectory plot."""
    fig, ax = plt.subplots(figsize=(14, 14))

    ax.imshow(
        Z,
        extent=(*scaled_eval_size[0], *scaled_eval_size[1]),
        origin="lower",
        cmap="jet",
        alpha=0.1,
        interpolation="bilinear",
    )
    cs = ax.contour(X, Y, Z, levels=20, cmap="jet")

    cbar = fig.colorbar(cs, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("f(x, y)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.plot(
        xs,
        ys,
        color="black",
        linewidth=2,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        label="Path",
    )

    ax.scatter(
        xs[0],
        ys[0],
        marker="o",
        s=80,
        color="lime",
        edgecolor="black",
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
        zorder=6,
        label="Final",
    )
    ax.scatter(
        global_minimums[:, 0],
        global_minimums[:, 1],
        marker="*",
        s=140,
        color="gold",
        edgecolor="black",
        zorder=7,
        label="Global Min",
    )

    ax.set_title(title_text, fontsize=12, pad=15)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_aspect("equal")

    # Legend
    legend_props = dict(frameon=True, framealpha=0.9, fontsize=9)
    main_legend = ax.legend(
        *zip(*dict(zip(*ax.get_legend_handles_labels())).items()),
        loc="upper left",
        **legend_props,
    )
    ax.add_artist(main_legend)

    # Metrics Legend
    active_metrics = sorted(
        [(k, v) for k, v in metrics.items() if isinstance(v, (int, float)) and v > 0],
        key=lambda item: item[1],
        reverse=True,
    )
    if active_metrics:
        metric_handles = [
            mpatches.Patch(
                color="none", label=f"{k.replace('_', ' ').title()}: {v:.4f}"
            )
            for k, v in active_metrics
        ]
        ax.legend(
            handles=metric_handles,
            loc="lower right",
            title="Evaluation Breakdown",
            title_fontsize=9,
            **legend_props,
        )

    out_path = out_dir / f"surface.{img_format}"
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return str(out_path)


def _save_dynamics_plot(
    out_dir: Path,
    z_vals: np.ndarray,
    grad_norms: np.ndarray,
    step_sizes: np.ndarray,
    displacement: np.ndarray,
    path_len: np.ndarray,
    func_name: str,
    img_format: str,
) -> str:
    """Save 2x2 dynamics analysis plot."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Optimization Dynamics | {func_name}", fontsize=14, y=0.96)
    steps = np.arange(len(z_vals))

    # 1. Objective Value
    ax = axs[0, 0]
    ax.plot(steps, z_vals, color=COLORS["loss"], linewidth=1.5)
    ax.set_yscale("symlog")
    _style_axis(ax, "Objective Value (SymLog)", ylabel="f(x)", log_scale=True)

    # 2. Gradient Norm
    ax = axs[0, 1]
    ax.plot(steps, grad_norms, color=COLORS["grad"], linewidth=1.5)
    ax.fill_between(steps, grad_norms, color=COLORS["grad"], alpha=0.1)
    _style_axis(ax, "Gradient Norm", ylabel="||∇f(x)||")

    # 3. Step Sizes
    ax = axs[1, 0]
    ax.plot(steps, step_sizes, color=COLORS["step"], linewidth=1.5)
    _style_axis(ax, "Step Sizes", xlabel="Iteration", ylabel="||Δx||")

    # 4. Efficiency
    ax = axs[1, 1]
    ax.plot(
        steps,
        path_len,
        color=COLORS["path"],
        label="Cumulative Path",
        linewidth=2,
        linestyle="--",
    )
    ax.plot(
        steps, displacement, color=COLORS["dist"], label="Net Displacement", linewidth=2
    )
    _style_axis(ax, "Trajectory Efficiency", xlabel="Iteration", ylabel="Distance")
    ax.legend(fontsize=8, frameon=False)

    # Efficiency Annotation
    if path_len[-1] > 0:
        eff = displacement[-1] / path_len[-1]
        ax.text(
            0.95,
            0.05,
            f"Efficiency: {eff:.2f}\n(1.0 = Straight Line)",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#ccc"),
        )

    out_path = out_dir / f"dynamics.{img_format}"
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return str(out_path)


def _save_phase_plot(
    out_dir: Path,
    grad_norms: np.ndarray,
    step_sizes: np.ndarray,
    func_name: str,
    img_format: str,
) -> str:
    """Save a Gradient Norm vs Step Size phase portrait."""
    fig, ax = plt.subplots(figsize=(9, 8))
    valid = (step_sizes > 1e-12) & (grad_norms > 1e-12)

    if np.any(valid):
        g_valid, s_valid, iters = (
            grad_norms[valid],
            step_sizes[valid],
            np.arange(len(grad_norms))[valid],
        )
        scatter = ax.scatter(
            g_valid,
            s_valid,
            c=iters,
            cmap="plasma",
            alpha=0.8,
            s=40,
            edgecolors="white",
            linewidth=0.5,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Iteration", rotation=270, labelpad=15, fontsize=9)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=10))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

    _style_axis(
        ax,
        f"Phase Portrait: Step vs Gradient | {func_name}",
        xlabel="Gradient Norm ||∇f(x)|| (Log)",
        ylabel="Step Size ||Δx|| (Log)",
        log_scale=True,
    )

    # Annotations
    props = dict(
        boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc", pad=0.5
    )
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # Stagnation (Bottom Right)
    ax.text(
        xlim[1] * 0.9,
        ylim[0] * 1.5,
        "STAGNATION\n(High Grad, No Move)",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#d62728",
        bbox=props,
    )
    # Overshooting (Top Left)
    ax.text(
        xlim[0] * 1.2,
        ylim[1] * 0.7,
        "OVERSHOOTING\n(Low Grad, Big Jump)",
        ha="left",
        va="top",
        fontsize=8,
        color="#ff7f0e",
        bbox=props,
    )
    # Convergence (Bottom Left)
    ax.text(
        xlim[0] * 1.2,
        ylim[0] * 1.5,
        "CONVERGENCE\n(Ideal)",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#2ca02c",
        bbox=props,
    )

    out_path = out_dir / f"phase_portrait.{img_format}"
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return str(out_path)


def _save_update_ratio_plot(
    out_dir: Path,
    grad_norms: np.ndarray,
    step_sizes: np.ndarray,
    func_name: str,
    img_format: str,
) -> str:
    """Save 'Update Efficiency Ratio' plot: Step Size / Gradient Norm."""
    fig, ax = plt.subplots(figsize=(12, 8))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = step_sizes / (grad_norms + 1e-10)

    ax.plot(np.arange(len(ratio)), ratio, color=COLORS["ratio"], linewidth=1.5)
    ax.set_yscale("log")
    _style_axis(
        ax,
        f"Effective Update Ratio (||Δx|| / ||∇f||) | {func_name}",
        xlabel="Iteration",
        ylabel="Ratio (Log Scale)",
        log_scale=True,
    )

    # Explanatory Annotations
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(
        len(ratio) * 0.02,
        1.3,
        "Aggressive / Momentum > 1.0",
        fontsize=8,
        color="#555555",
    )
    ax.text(
        len(ratio) * 0.02,
        0.7,
        "Conservative < 1.0",
        fontsize=8,
        color="#555555",
        va="top",
    )

    out_path = out_dir / f"update_ratio.{img_format}"
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return str(out_path)


def _save_penalty_donut(
    out_dir: Path,
    metrics: dict,
    func_name: str,
    img_format: str,
) -> str:
    """Save the tuning cost breakdown donut chart."""
    items = [
        (k, float(v))
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and v >= 0
    ]
    if not items:
        return ""

    keys, values = zip(*items)
    values = np.array(values)
    if np.all(values == 0):
        values = np.ones_like(values) * 1e-12

    use_log = (values.max() / (values[values > 0].min() + 1e-30)) > 500
    disp_vals = np.log1p(values) if use_log else values
    disp_vals[disp_vals <= 0] = 1e-12

    idx = np.argsort(disp_vals)[::-1]
    keys_sorted, vals_sorted = [keys[i] for i in idx], values[idx]
    colors = [METRIC_COLORS.get(k, "#bcbd22") for k in keys_sorted]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _ = ax.pie(
        disp_vals[idx],
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=1),
    )

    ax.text(0, 0, "Tuning\nObjectives", ha="center", va="center", fontweight="bold")
    ax.set_title(f"Tuning Cost Breakdown | {func_name}", fontsize=12, fontweight="bold")

    total = vals_sorted.sum() or 1.0
    labels = [
        f"{k.replace('_', ' ').title()}: {v:.4g} ({v / total * 100:.1f}%)"
        for k, v in zip(keys_sorted, vals_sorted)
    ]
    ax.legend(
        wedges,
        labels,
        title="Metrics",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9,
        frameon=False,
    )

    if use_log:
        plt.figtext(
            0.5,
            0.02,
            "Log-scaled for visibility.",
            ha="center",
            fontsize=8,
            color="#666",
        )

    out_path = out_dir / f"penalty_donut.{img_format}"
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return str(out_path)


def visualize_trajectory(
    func: Callable,
    func_name: str,
    cords: torch.Tensor,
    output_dir: str,
    optimizer_name: str,
    optimizer_params: dict,
    eval_metrics: dict,
    tune_metrics: dict,
    error_rate: float,
    global_minimums: torch.Tensor,
    eval_size: Tuple[Tuple[int, int], Tuple[int, int]],
    res: Union[int, str] = "auto",
    img_format: str = "png",
    debug: bool = False,
) -> Dict[str, str]:
    """Generates and saves a suite of visualization plots for an optimizer's trajectory.

    Creates a directory for the specific function and saves the following plots:
    1. Surface Plot: Contour map with the optimizer's path.
    2. Dynamics Plot: 2x2 grid of Loss, Gradient Norm, Step Sizes, and Efficiency.
    3. Phase Portrait: Step Size vs Gradient Norm (Log-Log) with regime annotations.
    4. Update Ratio: Plot of step size / gradient norm over time.
    5. Penalty Donut: Breakdown of the hyperparameter tuning objectives.

    Args:
        func: The objective function to evaluate.
        func_name: Name of the objective function (used for directory naming).
        cords: Tensor of shape [2, N] containing the optimization trajectory.
        output_dir: Directory where the function-specific folder will be created.
        optimizer_name: Name of the optimizer (used for titles).
        optimizer_params: Dictionary of hyperparameters used for the run.
        eval_metrics: Dictionary of metric values used for the final evaluation score.
        tune_metrics: Dictionary of metric values used during hyperparameter tuning.
        error_rate: The final weighted error score (evaluation metric).
        global_minimums: Tensor of shape [M, 2] containing known global minima.
        eval_size: Tuple ((min_x, max_x), (min_y, max_y)) defining the evaluation bounds.
        res: Resolution for the surface plot grid. Defaults to "auto".
        img_format: File format for saved images (e.g., "png", "jpg"). Defaults to "png".
        debug: If True, enables debug logging and disables caching.

    Returns:
        A dictionary mapping plot types ('surface', 'dynamics', 'phase_portrait',
        'update_ratio', 'penalty_donut') to their corresponding file paths.
    """
    func_dir = Path(output_dir)
    func_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Data
    pts = cords.t().detach().cpu().numpy()  # [N, 2]
    gm_pts = global_minimums.detach().cpu().numpy()
    xs, ys = pts[:, 0], pts[:, 1]

    # 2. Compute Surface
    scaled_eval_size = scale_eval_size(eval_size, 1.1)
    X, Y, Z = compute_surface(func, func_name, scaled_eval_size, res, debug=debug)
    X, Y, Z = X.numpy(), Y.numpy(), Z.numpy()

    # 3. Compute Metrics
    z_vals = _eval_z_on_trajectory(func, pts)
    grad_norms = _compute_gradient_norms(func, pts)
    displacement, path_len, step_sizes = _compute_efficiency(pts)

    # 4. Generate Main Title
    iterations = max(0, len(xs) - 1)
    config_str = ", ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in optimizer_params.items()
    )
    main_title = f"{func_name} | {optimizer_name}\nIters: {iterations}, Error: {error_rate:.4f}\n{config_str}"

    saved = {}

    saved["surface"] = _save_surface_plot(
        func_dir,
        X,
        Y,
        Z,
        scaled_eval_size,
        xs,
        ys,
        gm_pts,
        main_title,
        eval_metrics,
        img_format,
    )

    saved["dynamics"] = _save_dynamics_plot(
        func_dir,
        z_vals,
        grad_norms,
        step_sizes,
        displacement,
        path_len,
        func_name,
        img_format,
    )

    saved["phase_portrait"] = _save_phase_plot(
        func_dir, grad_norms, step_sizes, func_name, img_format
    )

    saved["update_ratio"] = _save_update_ratio_plot(
        func_dir, grad_norms, step_sizes, func_name, img_format
    )

    saved["penalty_donut"] = _save_penalty_donut(
        func_dir, tune_metrics, func_name, img_format
    )

    return saved
