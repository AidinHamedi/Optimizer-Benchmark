from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import (
    LogLocator,
    MaxNLocator,
    ScalarFormatter,
)

from .functions import scale_eval_size
from .utils.surface import compute_surface

VIS_SETTINGS = {
    "DPI": 120,
    "SIZES": {
        "SQUARE": (14, 14),
        "DYNAMICS": (14, 10),
        "PHASE": (18, 9),
        "WIDE": (8, 4),
        "DONUT": (8, 6),
    },
    "FONTS": {
        "TITLE": {"fontsize": 12, "fontweight": "bold"},
        "AXIS": {"fontsize": 10, "fontweight": "bold"},
        "LABEL_SIZE": 9,
        "TICK_SIZE": 8,
        "LEGEND_SIZE": 9,
        "ANNO": {"fontsize": 8, "fontweight": "medium"},
    },
    "COLORS": {
        "TEXT": "#222222",
        "TEXT_SEC": "#555555",
        "SPINE": "#888888",
        "GRID": "#dddddd",
        "LOSS": "#1f77b4",
        "GRAD": "#d62728",
        "STEP": "#2ca02c",
        "DIST": "#9467bd",
        "PATH": "#7f7f7f",
        "RATIO": "#e377c2",
        "EFF_FILL": "#9467bd",
    },
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


class PlotContext:
    """Context manager for creating, laying out, and saving matplotlib figures."""

    def __init__(
        self,
        path: Path,
        figsize: Tuple[int, int],
        layout_rect: Optional[list] = None,
        **kwargs,
    ):
        self.path = path
        self.layout_rect = layout_rect
        self.fig, self.axs = plt.subplots(figsize=figsize, **kwargs)
        self.ax = self.axs if not isinstance(self.axs, np.ndarray) else self.axs

    def __enter__(self):
        return self.fig, self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.layout_rect:
            plt.tight_layout(rect=self.layout_rect)
        else:
            plt.tight_layout()
        self.fig.savefig(self.path, bbox_inches="tight", dpi=VIS_SETTINGS["DPI"])
        plt.close(self.fig)


def _style_axis(
    ax: plt.Axes, title: str, xlabel: str = None, ylabel: str = None, log: bool = False
):
    """Applies standardized styling to a matplotlib axis."""
    c, f = VIS_SETTINGS["COLORS"], VIS_SETTINGS["FONTS"]

    ax.set_title(title, pad=8, color=c["TEXT"], **f["TITLE"])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=f["LABEL_SIZE"], color=c["TEXT_SEC"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=f["LABEL_SIZE"], color=c["TEXT_SEC"])

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(c["SPINE"])

    ax.tick_params(axis="both", colors=c["TEXT_SEC"], labelsize=f["TICK_SIZE"])

    if log:
        ax.grid(True, which="major", ls="-", alpha=0.5, color=c["GRID"])
        ax.grid(True, which="minor", ls=":", alpha=0.3, color=c["GRID"])
    else:
        ax.grid(True, ls="--", alpha=0.5, color=c["GRID"])


def _ema_smooth(data: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Computes Exponential Moving Average for a sequence."""
    if len(data) < 2:
        return data
    smoothed = np.empty_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def _compute_trajectory_data(
    func: Callable, pts: np.ndarray, debug: bool = False
) -> Dict[str, np.ndarray]:
    """Calculates loss, gradients, steps, and efficiency metrics from trajectory points."""
    pts_tensor = torch.from_numpy(pts).float()

    if pts.size == 0:
        raise ValueError("Trajectory is empty.")

    try:
        z_vals = func(pts_tensor).detach().numpy().ravel()
        if z_vals.size != pts.shape[0]:
            raise RuntimeError("Batch eval size mismatch")
    except Exception:
        z_list = []
        for p in pts_tensor:
            try:
                z_list.append(func(p).item())
            except Exception:
                z_list.append(func(p.unsqueeze(0)).item())
        z_vals = np.array(z_list)

    grad_norms = []
    for p in pts_tensor:
        p = p.detach().requires_grad_(True)
        try:
            val = func(p)
            if val.numel() > 1:
                val = val.sum()
            val.backward()
            grad_norms.append(p.grad.norm().item() if p.grad is not None else 0.0)
        except Exception:
            grad_norms.append(0.0)
    grad_norms = np.array(grad_norms)

    # Calculate steps. Append 0.0 for the final point (stop state)
    step_sizes = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    step_sizes = np.append(step_sizes, 0.0)

    path_len = np.cumsum(step_sizes)
    displacement = np.linalg.norm(pts - pts[0], axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        eff_index = displacement / (path_len + 1e-12)
        eff_index[0] = 1.0

    if debug:
        print(f"[Vis] Trajectory: {len(pts)} points")
        print(f"[Vis] Z Range: [{z_vals.min():.2e}, {z_vals.max():.2e}]")
        print(f"[Vis] Grad Range: [{grad_norms.min():.2e}, {grad_norms.max():.2e}]")
        print(f"[Vis] Step Range: [{step_sizes.min():.2e}, {step_sizes.max():.2e}]")

    return {
        "z": z_vals,
        "grads": grad_norms,
        "steps": step_sizes,
        "path_eff": eff_index,
        "path": path_len,
        "dist": displacement,
        "xs": pts[:, 0],
        "ys": pts[:, 1],
    }


def _plot_surface(
    out_path: Path,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    data: Dict[str, np.ndarray],
    bounds: Tuple,
    gm: np.ndarray,
    title: str,
    metrics: Dict,
):
    """Draws the function surface contour and the optimizer's path."""
    c, f = VIS_SETTINGS["COLORS"], VIS_SETTINGS["FONTS"]

    # Detect plateau & choose scale
    z_min, z_max = Z.min(), Z.max()
    z_range = z_max - z_min + 1e-12
    low_frac = np.mean(Z < z_min + 0.15 * z_range)

    use_log = low_frac > 0.3 and z_range > 1e-6 and z_min > 1e-12

    if use_log:
        norm = mcolors.LogNorm(vmin=z_min, vmax=z_max)
        levels = np.geomspace(z_min, z_max, 20)
    else:
        norm, levels = None, 20

    with PlotContext(out_path, figsize=VIS_SETTINGS["SIZES"]["SQUARE"]) as (fig, ax):
        ax.imshow(
            Z,
            extent=(*bounds[0], *bounds[1]),
            origin="lower",
            cmap="jet",
            alpha=0.1,
            norm=norm,
        )
        cs = ax.contour(X, Y, Z, levels=levels, cmap="jet", norm=norm)

        cbar = fig.colorbar(cs, ax=ax, shrink=0.8, pad=0.02)

        if use_log:

            def log_fmt(x, _):
                if x >= 1:
                    return f"{x:.1f}"
                if x >= 0.01:
                    return f"{x:.2f}"
                return f"{x:.1e}"

            cbar.ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
            cbar.ax.yaxis.set_minor_locator(
                LogLocator(base=10, subs="auto", numticks=12)
            )
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(log_fmt))
        else:
            cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            cbar.ax.yaxis.get_major_formatter().set_powerlimits((-2, 3))
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

        for val, color, label in [
            (data["z"][0], "lime", "S"),
            (data["z"][-1], "red", "F"),
        ]:
            if z_min <= val <= z_max:
                cbar.ax.axhline(val, color=color, lw=2.5, alpha=0.9)
                cbar.ax.text(
                    -0.2,
                    val,
                    label,
                    color=color,
                    fontsize=f["LABEL_SIZE"],
                    fontweight="bold",
                    va="center",
                    ha="right",
                    transform=cbar.ax.get_yaxis_transform(),
                )

        cbar.set_label(
            f"f(x,y) [{'Log' if use_log else 'Linear'}]",
            fontsize=f["LABEL_SIZE"],
            fontweight="bold",
        )
        cbar.ax.tick_params(labelsize=f["TICK_SIZE"])

        ax.plot(
            data["xs"],
            data["ys"],
            color="black",
            lw=2,
            marker="o",
            ms=5,
            mfc="white",
            label="Path",
        )
        ax.scatter(
            data["xs"][0],
            data["ys"][0],
            s=80,
            c="lime",
            ec="black",
            zorder=5,
            label="Start",
        )
        ax.scatter(
            data["xs"][-1],
            data["ys"][-1],
            s=110,
            c="red",
            ec="black",
            zorder=6,
            label="Final",
        )
        ax.scatter(
            gm[:, 0],
            gm[:, 1],
            s=140,
            marker="*",
            c="gold",
            ec="black",
            zorder=7,
            label="Global Min",
        )

        ax.set_title(title, color=c["TEXT"], **f["TITLE"])
        ax.set_aspect("equal")

        leg1 = ax.legend(loc="upper left", fontsize=f["LEGEND_SIZE"])

        sorted_metrics = sorted(
            [(k, v) for k, v in metrics.items() if v > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        if sorted_metrics:
            patches = [
                mpatches.Patch(
                    color="none", label=f"{k.replace('_', ' ').title()}: {v:.4f}"
                )
                for k, v in sorted_metrics
            ]
            ax.legend(
                handles=patches,
                loc="lower right",
                title="Evaluation Breakdown",
                fontsize=f["LEGEND_SIZE"],
            )
            ax.add_artist(leg1)


def _plot_dynamics(out_path: Path, data: Dict[str, np.ndarray], name: str):
    """Draws time-series metrics: Loss, Gradients, Steps, and Efficiency."""
    c, f = VIS_SETTINGS["COLORS"], VIS_SETTINGS["FONTS"]

    with PlotContext(
        out_path,
        figsize=VIS_SETTINGS["SIZES"]["DYNAMICS"],
        layout_rect=[0, 0.03, 1, 0.95],
        nrows=2,
        ncols=2,
    ) as (fig, axs):
        fig.suptitle(
            f"Optimization Dynamics | {name}",
            y=0.98,
            color=c["TEXT"],
            **f["TITLE"],
        )
        steps = np.arange(len(data["z"]))

        # Objective
        ax = axs[0, 0]
        ax.plot(steps, data["z"], color=c["LOSS"])
        ax.set_yscale("symlog")
        _style_axis(ax, "Objective Value (SymLog)", ylabel="f(x)", log=True)

        # Gradients
        ax = axs[0, 1]
        ax.plot(steps, data["grads"], color=c["GRAD"], alpha=0.8, lw=1.5)
        ax.fill_between(steps, data["grads"], color=c["GRAD"], alpha=0.1)
        _style_axis(ax, "Gradient Norm", ylabel="||∇f(x)||")

        # Steps
        ax = axs[1, 0]
        ax.plot(steps, data["steps"], color=c["STEP"], alpha=0.8, lw=1.5)
        _style_axis(ax, "Step Sizes", xlabel="Iteration", ylabel="||Δx||")

        # Efficiency
        ax = axs[1, 1]
        ax.fill_between(
            steps,
            0,
            data["path_eff"],
            color=c["EFF_FILL"],
            alpha=0.15,
            label="Efficiency Ratio",
        )
        _style_axis(
            ax,
            "Trajectory Efficiency (Disp / Path)",
            xlabel="Iteration",
            ylabel="Efficiency (0-1)",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.label.set_color(c["EFF_FILL"])
        ax.tick_params(axis="y", colors=c["EFF_FILL"])

        ax2 = ax.twinx()
        ax2.plot(
            steps,
            data["path"],
            color=c["PATH"],
            ls=":",
            alpha=0.8,
            lw=2,
            label="Total Path (Cost)",
        )
        ax2.plot(
            steps,
            data["dist"],
            color=c["DIST"],
            ls="-",
            alpha=0.9,
            lw=2.5,
            label="Net Displacement (Gain)",
        )

        ax2.set_ylabel("Distance", color=c["TEXT_SEC"], fontsize=f["LABEL_SIZE"])
        ax2.tick_params(axis="y", labelcolor=c["TEXT_SEC"], labelsize=f["TICK_SIZE"])
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(c["SPINE"])

        ax.legend(
            ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0],
            ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1],
            loc="lower right",
            fontsize=f["LEGEND_SIZE"],
        )


def _plot_phase_portrait(out_path: Path, data: Dict[str, np.ndarray], name: str):
    """
    Draws Side-by-Side Phase Portrait (Step vs Grad).
    Left: Raw Jitter. Right: Smoothed Trend.
    """
    c, f = VIS_SETTINGS["COLORS"], VIS_SETTINGS["FONTS"]

    fig = plt.figure(figsize=VIS_SETTINGS["SIZES"]["PHASE"])
    gs = GridSpec(
        1,
        3,
        width_ratios=[1.0, 1.0, 0.05],
        wspace=0.15,
        figure=fig,
    )

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    try:
        x_raw, y_raw = data["grads"][:-1], data["steps"][:-1]
        iters = np.arange(len(x_raw))

        epsilon = 1e-14
        mask = (x_raw > epsilon) & (y_raw > epsilon)
        x_cl, y_cl = x_raw[mask], y_raw[mask]
        iters_cl = iters[mask]

        x_sm = _ema_smooth(x_cl, alpha=0.15)
        y_sm = _ema_smooth(y_cl, alpha=0.15)

        if len(x_cl) > 0:
            all_v = np.concatenate([x_cl, y_cl, x_sm, y_sm])
            l_min = (all_v.min() + epsilon) * 0.5
            l_max = all_v.max() * 2.0
        else:
            l_min, l_max = 1e-5, 1.0

        def draw_phase(ax, xd, yd, title, smooth=False):
            ax.plot(
                [l_min, l_max],
                [l_min, l_max],
                color="#cccccc",
                ls="--",
                lw=1,
                zorder=0,
            )

            lc_obj = None
            if len(xd) > 2:
                points = np.array([xd, yd]).T.reshape(-1, 1, 2)
                segs = np.concatenate([points[:-1], points[1:]], axis=1)

                lc_obj = mcoll.LineCollection(
                    segs,
                    cmap="plasma",
                    norm=plt.Normalize(0, len(xd)),
                    alpha=0.9 if smooth else 0.6,
                    lw=2.0 if smooth else 1.0,
                    zorder=2,
                )
                lc_obj.set_array(iters_cl)
                ax.add_collection(lc_obj)

                ax.scatter(
                    xd,
                    yd,
                    c=iters_cl,
                    cmap="plasma",
                    alpha=0.1 if smooth else 0.5,
                    s=5 if smooth else 15,
                    zorder=3,
                )

                ax.scatter(
                    xd[0],
                    yd[0],
                    c="lime",
                    s=100,
                    ec="black",
                    zorder=10,
                    label="Start",
                )
                ax.scatter(
                    xd[-1],
                    yd[-1],
                    c="red",
                    s=100,
                    ec="black",
                    zorder=10,
                    label="End",
                )
            else:
                ax.scatter(xd, yd, alpha=0.5)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(l_min, l_max)
            ax.set_ylim(l_min, l_max)
            ax.set_aspect("equal", adjustable="box")

            _style_axis(ax, title, "Gradient Norm (Log)", "Step Size (Log)", log=True)

            bbox = dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.85,
                edgecolor="#eeeeee",
            )
            props = {**f["ANNO"], "bbox": bbox}

            ax.text(
                l_max * 0.5,
                l_min * 2,
                "STAGNATION",
                ha="right",
                va="bottom",
                color=c["GRAD"],
                **props,
            )
            ax.text(
                l_min * 2,
                l_max * 0.5,
                "INSTABILITY",
                ha="left",
                va="top",
                color=c["DIST"],
                **props,
            )

            return lc_obj

        draw_phase(
            ax_left,
            x_cl,
            y_cl,
            f"Raw Dynamics (Jitter) | {name}",
            smooth=False,
        )
        lc = draw_phase(
            ax_right,
            x_sm,
            y_sm,
            f"Smoothed Trend (Flow) | {name}",
            smooth=True,
        )

        items = [
            mpatches.Patch(color="purple", label="Trajectory (Time)"),
            plt.Line2D(
                [0],
                [0],
                color="#cccccc",
                lw=1,
                ls="--",
                label="Ratio = 1.0",
            ),
        ]
        ax_right.legend(
            handles=items,
            loc="upper right",
            fontsize=f["LEGEND_SIZE"],
        )

        if lc:
            cb = fig.colorbar(lc, cax=cax)
            cb.set_label(
                "Iteration",
                rotation=270,
                labelpad=15,
                size=f["LABEL_SIZE"],
            )
            cb.ax.tick_params(labelsize=f["TICK_SIZE"])

        fig.savefig(out_path, dpi=VIS_SETTINGS["DPI"], bbox_inches="tight")

    finally:
        plt.close(fig)


def _plot_update_ratio(out_path: Path, data: Dict[str, np.ndarray], name: str):
    """Draws Step/Gradient ratio over time."""
    c, f = VIS_SETTINGS["COLORS"], VIS_SETTINGS["FONTS"]

    with PlotContext(out_path, figsize=VIS_SETTINGS["SIZES"]["WIDE"]) as (fig, ax):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = data["steps"] / (data["grads"] + 1e-10)

        ax.plot(ratio, color=c["RATIO"], lw=1)
        ax.set_yscale("log")
        _style_axis(
            ax,
            f"Effective Update Ratio (||Δx|| / ||∇f||) | {name}",
            "Iteration",
            "Ratio (Log Scale)",
            log=True,
        )

        ax.axhline(1.0, c=c["TEXT_SEC"], ls="--", alpha=0.5)
        ax.text(
            len(ratio) * 0.02,
            1.2,
            "Aggressive > 1.0",
            color=c["TEXT_SEC"],
            **f["ANNO"],
        )


def _plot_penalty_donut(out_path: Path, metrics: Dict, name: str):
    """Draws a donut chart of tuning penalties."""
    c, f = VIS_SETTINGS["COLORS"], VIS_SETTINGS["FONTS"]

    with PlotContext(out_path, figsize=VIS_SETTINGS["SIZES"]["DONUT"]) as (fig, ax):
        valid = [(k, float(v)) for k, v in metrics.items() if v > 0]
        if not valid:
            ax.text(
                0.5, 0.5, "No tuning metrics", ha="center", fontsize=f["LABEL_SIZE"]
            )
            ax.axis("off")
            return

        keys, vals = zip(*sorted(valid, key=lambda x: x[1], reverse=True))
        vals = np.array(vals)
        disp_vals = np.log1p(vals) if (vals.max() / (vals.min() + 1e-9) > 500) else vals

        wedges, _ = ax.pie(
            disp_vals,
            startangle=90,
            colors=[METRIC_COLORS.get(k, "#bcbd22") for k in keys],
            wedgeprops=dict(width=0.4, edgecolor="white"),
        )

        ax.text(
            0,
            0,
            "Tuning\nObjectives",
            ha="center",
            va="center",
            color=c["TEXT"],
            **f["AXIS"],
        )
        ax.set_title(f"Tuning Cost Breakdown | {name}", color=c["TEXT"], **f["TITLE"])

        labels = [
            f"{k.replace('_', ' ').title()}: {v:.4g} ({v / vals.sum() * 100:.1f}%)"
            for k, v in zip(keys, vals)
        ]
        ax.legend(
            wedges,
            labels,
            title="Metrics",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=f["LEGEND_SIZE"],
        )

        if vals.max() / (vals.min() + 1e-9) > 500:
            plt.figtext(
                0.5,
                0.02,
                "Log-scaled for visibility.",
                ha="center",
                fontsize=f["ANNO"]["fontsize"],
                color=c["TEXT_SEC"],
            )


def visualize_trajectory(
    func: Callable,
    func_name: str,
    cords: torch.Tensor,
    output_dir: str,
    optimizer_name: str,
    optimizer_params: Dict,
    eval_metrics: Dict,
    tune_metrics: Dict,
    error_rate: float,
    global_minimums: torch.Tensor,
    eval_size: Tuple[Tuple[int, int], Tuple[int, int]],
    res: Any = "auto",
    img_format: str = "png",
    debug: bool = False,
) -> Dict[str, str]:
    """
    Generates and saves a suite of visualization plots for an optimizer's trajectory.

    Args:
        func: Objective function.
        func_name: Name of the function.
        cords: Tensor of trajectory points [2, N].
        output_dir: Root directory for output.
        optimizer_name: Name of the optimizer.
        optimizer_params: Hyperparameters used.
        eval_metrics: Final evaluation metrics.
        tune_metrics: Tuning cost breakdown.
        error_rate: Final error score.
        global_minimums: Coordinates of global minima.
        eval_size: Bounds for the surface plot.
        res: Resolution for surface plot.
        img_format: Image file extension.
        debug: Enable debug output.

    Returns:
        Dictionary mapping plot types to file paths.
    """
    if debug:
        print(
            f"[Vis] Generating visualization for {func_name} with {optimizer_name}..."
        )

    func_dir = Path(output_dir)
    func_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Data
    pts = cords.t().detach().cpu().numpy()
    gm_pts = global_minimums.detach().cpu().numpy()
    traj_data = _compute_trajectory_data(func, pts, debug=debug)

    # 2. Compute Surface
    scaled_bounds = scale_eval_size(eval_size, 1.1)
    X, Y, Z = compute_surface(func, func_name, scaled_bounds, res, debug=debug)
    X, Y, Z = X.numpy(), Y.numpy(), Z.numpy()

    # 3. Generate Title
    param_str = ", ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in optimizer_params.items()
    )
    title = f"{func_name} | {optimizer_name}\nIters: {len(pts) - 1}, Error: {error_rate:.4f}\n{param_str}"

    # 4. Generate Plots
    files = {}

    f_surf = func_dir / f"surface.{img_format}"
    _plot_surface(
        f_surf, X, Y, Z, traj_data, scaled_bounds, gm_pts, title, eval_metrics
    )
    files["surface"] = str(f_surf)

    f_dyn = func_dir / f"dynamics.{img_format}"
    _plot_dynamics(f_dyn, traj_data, func_name)
    files["dynamics"] = str(f_dyn)

    f_phase = func_dir / f"phase_portrait.{img_format}"
    _plot_phase_portrait(f_phase, traj_data, func_name)
    files["phase_portrait"] = str(f_phase)

    f_ratio = func_dir / f"update_ratio.{img_format}"
    _plot_update_ratio(f_ratio, traj_data, func_name)
    files["update_ratio"] = str(f_ratio)

    f_donut = func_dir / f"penalty_donut.{img_format}"
    _plot_penalty_donut(f_donut, tune_metrics, func_name)
    files["penalty_donut"] = str(f_donut)

    return files
