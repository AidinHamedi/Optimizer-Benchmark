from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, MaxNLocator, ScalarFormatter

from .functions import scale_eval_size
from .utils.surface import compute_surface


@dataclass(frozen=True)
class PlotSizes:
    square: Tuple[int, int] = (14, 14)
    dynamics: Tuple[int, int] = (14, 10)
    phase: Tuple[int, int] = (18, 9)
    wide: Tuple[int, int] = (8, 4)
    donut: Tuple[int, int] = (8, 6)


@dataclass(frozen=True)
class FontStyles:
    title: Dict[str, Any] = field(
        default_factory=lambda: {"fontsize": 12, "fontweight": "bold"}
    )
    axis: Dict[str, Any] = field(
        default_factory=lambda: {"fontsize": 10, "fontweight": "bold"}
    )
    annotation: Dict[str, Any] = field(
        default_factory=lambda: {"fontsize": 8, "fontweight": "medium"}
    )
    label_size: int = 9
    tick_size: int = 8
    legend_size: int = 9


@dataclass(frozen=True)
class PlotColors:
    text_primary: str = "#222222"
    text_secondary: str = "#555555"
    spine: str = "#888888"
    grid: str = "#dddddd"
    loss_curve: str = "#1f77b4"
    gradient_curve: str = "#d62728"
    step_curve: str = "#2ca02c"
    displacement_curve: str = "#9467bd"
    path_curve: str = "#7f7f7f"
    ratio_curve: str = "#e377c2"
    efficiency_fill: str = "#9467bd"
    start_point: str = "lime"
    end_point: str = "red"
    global_min: str = "gold"
    phase_background: str = "#cccccc"

    # Metric specific colors for donut chart
    metric_val_penalty: str = "#1f77b4"
    metric_dist_penalty: str = "#ff7f0e"
    metric_speed_penalty: str = "#2ca02c"
    metric_eff_penalty: str = "#d62728"
    metric_bound_penalty: str = "#9467bd"
    metric_terrain_violation: str = "#8c564b"
    metric_jump_penalty: str = "#e377c2"
    metric_prox_penalty: str = "#7f7f7f"
    metric_start_prox_penalty: str = "#7f7f7f"

    def get_metric_color(self, key: str) -> str:
        """Dynamically retrieve color for a tuning metric based on its key."""
        attr_name = f"metric_{key}"
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        return "#bcbd22"  # Fallback color


@dataclass(frozen=True)
class MarkerStyles:
    start_size: int = 80
    end_size: int = 110
    global_min_size: int = 140
    path_width: int = 2
    path_dot_size: int = 5


@dataclass(frozen=True)
class CalculationParams:
    efficiency_threshold: float = 1.4
    ema_smoothing_factor: float = 0.15
    surface_log_threshold: float = 0.3
    surface_padding_factor: float = 1.1


@dataclass(frozen=True)
class VisualizationSettings:
    dpi: int = 120
    sizes: PlotSizes = field(default_factory=PlotSizes)
    fonts: FontStyles = field(default_factory=FontStyles)
    colors: PlotColors = field(default_factory=PlotColors)
    markers: MarkerStyles = field(default_factory=MarkerStyles)
    params: CalculationParams = field(default_factory=CalculationParams)


# Global Configuration Instance
VIS_CONFIG = VisualizationSettings()


@dataclass
class TrajectoryData:
    loss_values: np.ndarray
    gradient_norms: np.ndarray
    step_sizes: np.ndarray
    efficiency_scores: np.ndarray
    cumulative_path: np.ndarray
    displacement: np.ndarray
    xs: np.ndarray
    ys: np.ndarray


class FigureContext:
    """
    Context manager for handling matplotlib figure lifecycles.
    Ensures figures are closed properly to prevent memory leaks and handles layout.
    """

    def __init__(
        self,
        output_path: Path,
        figsize: Tuple[int, int],
        layout_rect: Optional[List[float]] = None,
        dpi: int = 120,
        **subplots_kwargs,
    ):
        self.output_path = output_path
        self.layout_rect = layout_rect
        self.dpi = dpi
        self.fig, self.axs = plt.subplots(figsize=figsize, **subplots_kwargs)
        # Standardize access to axes (single vs array)
        self.ax = self.axs if not isinstance(self.axs, np.ndarray) else self.axs

    def __enter__(self):
        return self.fig, self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            plt.close(self.fig)
            return False  # Propagate exception

        if self.layout_rect:
            plt.tight_layout(rect=self.layout_rect)
        else:
            plt.tight_layout()

        self.fig.savefig(self.output_path, bbox_inches="tight", dpi=self.dpi)
        plt.close(self.fig)
        return True


def compute_ema(data: np.ndarray, alpha: float) -> np.ndarray:
    """Computes Exponential Moving Average (EMA) for smoothing noisy series."""
    if len(data) < 2:
        return data
    smoothed = np.empty_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def compute_spatial_efficiency(
    points: np.ndarray,
    step_sizes: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Computes a spatial efficiency score based on the Relative Scale Gate model."""
    n_points = len(points)
    efficiency = np.ones(n_points, dtype=float)

    # Use tensors for vectorized logic consistent with criterion.py
    points_t = torch.from_numpy(points).float().T  # [2, N]
    steps_t = torch.from_numpy(step_sizes).float()  # [N]

    for i in range(1, n_points):
        # Subset of trajectory up to current point
        sub_points = points_t[:, : i + 1]
        sub_steps = steps_t[:i]

        # 1. Calculate Footprint (Span)
        bbox_max = torch.max(sub_points, dim=1).values
        bbox_min = torch.min(sub_points, dim=1).values
        span = torch.norm(bbox_max - bbox_min)

        # 2. Filter Jitter (1% of peak velocity in current window)
        max_s = torch.max(sub_steps)
        active_mask = sub_steps > (max_s * 0.01)
        significant_effort = torch.sum(sub_steps[active_mask])

        # 3. Compute Efficiency Ratio
        raw_eff = (span * threshold) / (significant_effort + 1e-12)
        efficiency[i] = torch.clamp(raw_eff, max=1.0).item()

    return efficiency


class OptimizerVisualizer:
    """
    Handles the generation of all visualization plots for a specific optimizer run.
    Encapsulates styling, data computation, and plotting logic.
    """

    def __init__(
        self,
        output_dir: Path,
        config: VisualizationSettings = VIS_CONFIG,
        debug: bool = False,
    ):
        self.output_dir = output_dir
        self.config = config
        self.debug = debug

    def compute_metrics(self, func: Callable, points: np.ndarray) -> TrajectoryData:
        """Derives physical and gradient-based metrics from coordinate trajectory."""
        points_tensor = torch.from_numpy(points).float()

        if points.size == 0:
            raise ValueError("Trajectory is empty.")

        # 1. Compute Loss Values (Forward Pass)
        try:
            z_vals = func(points_tensor).detach().numpy().ravel()
            if z_vals.size != points.shape[0]:
                raise RuntimeError("Batch eval size mismatch")
        except Exception:
            # Fallback for functions that don't support batch processing
            z_list = []
            for p in points_tensor:
                try:
                    z_list.append(func(p).item())
                except Exception:
                    z_list.append(func(p.unsqueeze(0)).item())
            z_vals = np.array(z_list)

        # 2. Compute Gradients (Backward Pass)
        grad_norms = []
        for p in points_tensor:
            p = p.detach().requires_grad_(True)
            try:
                val = func(p)
                if val.numel() > 1:
                    val = val.sum()
                val.backward()
                norm = p.grad.norm().item() if p.grad is not None else 0.0
                grad_norms.append(norm)
            except Exception:
                grad_norms.append(0.0)
        grad_norms = np.array(grad_norms)

        # 3. Kinematics (Step sizes, Path length)
        diffs = points[1:] - points[:-1]
        step_sizes = np.linalg.norm(diffs, axis=1)
        step_sizes = np.append(step_sizes, 0.0)  # Append 0 to match length

        cumulative_path = np.cumsum(step_sizes)
        displacement = np.linalg.norm(points - points[0], axis=1)

        # 4. Efficiency
        efficiency = compute_spatial_efficiency(
            points,
            step_sizes,
            self.config.params.efficiency_threshold,
        )

        if self.debug:
            print(f"[Vis] Trajectory: {len(points)} points")
            print(f"[Vis] Z Range: [{z_vals.min():.2e}, {z_vals.max():.2e}]")

        return TrajectoryData(
            loss_values=z_vals,
            gradient_norms=grad_norms,
            step_sizes=step_sizes,
            efficiency_scores=efficiency,
            cumulative_path=cumulative_path,
            displacement=displacement,
            xs=points[:, 0],
            ys=points[:, 1],
        )

    def style_axis(
        self,
        ax: plt.Axes,
        title: str,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        log_grid: bool = False,
    ):
        """Applies consistent styling to a matplotlib axis."""
        ax.set_title(
            title,
            pad=8,
            color=self.config.colors.text_primary,
            **self.config.fonts.title,
        )

        if xlabel:
            ax.set_xlabel(
                xlabel,
                fontsize=self.config.fonts.label_size,
                color=self.config.colors.text_secondary,
            )
        if ylabel:
            ax.set_ylabel(
                ylabel,
                fontsize=self.config.fonts.label_size,
                color=self.config.colors.text_secondary,
            )

        # Spine visibility
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(self.config.colors.spine)

        ax.tick_params(
            axis="both",
            colors=self.config.colors.text_secondary,
            labelsize=self.config.fonts.tick_size,
        )

        # Grid styling
        grid_color = self.config.colors.grid
        if log_grid:
            ax.grid(True, which="major", ls="-", alpha=0.5, color=grid_color)
            ax.grid(True, which="minor", ls=":", alpha=0.3, color=grid_color)
        else:
            ax.grid(True, ls="--", alpha=0.5, color=grid_color)

    def add_trajectory_markers(
        self,
        ax: plt.Axes,
        xs: np.ndarray,
        ys: np.ndarray,
        gm: Optional[np.ndarray] = None,
    ):
        """Adds Start (Green), End (Red), and Global Minimum (Gold Star) markers."""
        ax.scatter(
            xs[0],
            ys[0],
            s=self.config.markers.start_size,
            c=self.config.colors.start_point,
            ec="black",
            zorder=5,
            label="Start",
        )
        ax.scatter(
            xs[-1],
            ys[-1],
            s=self.config.markers.end_size,
            c=self.config.colors.end_point,
            ec="black",
            zorder=6,
            label="End",
        )
        if gm is not None:
            ax.scatter(
                gm[:, 0],
                gm[:, 1],
                s=self.config.markers.global_min_size,
                marker="*",
                c=self.config.colors.global_min,
                ec="black",
                zorder=7,
                label="Global Min",
            )

    def plot_surface(
        self,
        filename: str,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        data: TrajectoryData,
        bounds: Tuple,
        gm: np.ndarray,
        title: str,
        eval_metrics: Dict,
    ):
        """Generates the 2D Contour/Surface plot with the optimization path overlay."""
        path = self.output_dir / filename

        # Determine if Log scale is needed for the contour map
        z_min, z_max = Z.min(), Z.max()
        z_range = z_max - z_min + 1e-12

        low_val_fraction = np.mean(Z < z_min + 0.15 * z_range)
        use_log_scale = (
            low_val_fraction > self.config.params.surface_log_threshold
            and z_range > 1e-6
        )

        if use_log_scale:
            norm = mcolors.LogNorm(vmin=z_min, vmax=z_max)
            levels = np.geomspace(z_min, z_max, 20)
        else:
            norm, levels = None, 20

        with FigureContext(path, self.config.sizes.square, dpi=self.config.dpi) as (
            fig,
            ax,
        ):
            # Background heatmap
            ax.imshow(
                Z,
                extent=(*bounds[0], *bounds[1]),
                origin="lower",
                cmap="jet",
                alpha=0.1,
                norm=norm,
            )
            # Contour lines
            cs = ax.contour(X, Y, Z, levels=levels, cmap="jet", norm=norm)

            # Colorbar configuration
            cbar = fig.colorbar(cs, ax=ax, shrink=0.8, pad=0.02)

            if use_log_scale:
                cbar.ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
                cbar.ax.yaxis.set_minor_locator(
                    LogLocator(base=10, subs="auto", numticks=12)
                )
                cbar.ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(
                        lambda x, _: f"{x:.1f}"
                        if x >= 1
                        else (f"{x:.2f}" if x >= 0.01 else f"{x:.1e}")
                    )
                )
            else:
                cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                cbar.ax.yaxis.get_major_formatter().set_powerlimits((-2, 3))
                cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

            # Add markers to colorbar for Start/End values
            for val, color, label in [
                (data.loss_values[0], self.config.colors.start_point, "S"),
                (data.loss_values[-1], self.config.colors.end_point, "F"),
            ]:
                if z_min <= val <= z_max:
                    cbar.ax.axhline(val, color=color, lw=2.5, alpha=0.9)
                    cbar.ax.text(
                        -0.2,
                        val,
                        label,
                        color=color,
                        fontsize=self.config.fonts.label_size,
                        fontweight="bold",
                        va="center",
                        ha="right",
                        transform=cbar.ax.get_yaxis_transform(),
                    )

            cbar.set_label(
                f"f(x,y) [{'Log' if use_log_scale else 'Linear'}]",
                **self.config.fonts.axis,
            )
            cbar.ax.tick_params(labelsize=self.config.fonts.tick_size)

            # Plot Trajectory Path
            ax.plot(
                data.xs,
                data.ys,
                color="black",
                lw=self.config.markers.path_width,
                marker="o",
                ms=self.config.markers.path_dot_size,
                mfc="white",
                label="Path",
            )
            self.add_trajectory_markers(ax, data.xs, data.ys, gm)

            ax.set_title(
                title, color=self.config.colors.text_primary, **self.config.fonts.title
            )
            ax.set_aspect("equal")

            # Main Legend
            leg1 = ax.legend(loc="upper left", fontsize=self.config.fonts.legend_size)

            # Secondary Legend: Metrics Breakdown
            valid_metrics = sorted(
                [(k, v) for k, v in eval_metrics.items() if v > 0],
                key=lambda x: x[1],
                reverse=True,
            )
            if valid_metrics:
                patches = [
                    mpatches.Patch(
                        color="none", label=f"{k.replace('_', ' ').title()}: {v:.4f}"
                    )
                    for k, v in valid_metrics
                ]
                ax.legend(
                    handles=patches,
                    loc="lower right",
                    title="Evaluation Breakdown",
                    fontsize=self.config.fonts.legend_size,
                )
                ax.add_artist(leg1)

    def plot_dynamics(self, filename: str, data: TrajectoryData, func_name: str):
        """Plots optimization metrics (Loss, Gradient, Step Size, Efficiency) over time."""
        path = self.output_dir / filename

        with FigureContext(
            path,
            self.config.sizes.dynamics,
            layout_rect=[0, 0.03, 1, 0.95],
            dpi=self.config.dpi,
            nrows=2,
            ncols=2,
        ) as (fig, axs):
            fig.suptitle(
                f"Optimization Dynamics: {func_name}",
                y=0.98,
                color=self.config.colors.text_primary,
                **self.config.fonts.title,
            )
            steps = np.arange(len(data.loss_values))

            # 1. Loss Function
            ax = axs[0, 0]
            ax.plot(steps, data.loss_values, color=self.config.colors.loss_curve)
            ax.set_yscale("symlog")
            self.style_axis(ax, "Loss Function", ylabel=r"$f(x)$", log_grid=True)

            # 2. Gradient Magnitude
            ax = axs[0, 1]
            ax.plot(
                steps,
                data.gradient_norms,
                color=self.config.colors.gradient_curve,
                alpha=0.8,
                lw=1.5,
            )
            ax.fill_between(
                steps,
                data.gradient_norms,
                color=self.config.colors.gradient_curve,
                alpha=0.1,
            )
            self.style_axis(ax, "Gradient Magnitude", ylabel=r"$||\nabla f||$")

            # 3. Step Sizes
            ax = axs[1, 0]
            ax.plot(
                steps,
                data.step_sizes,
                color=self.config.colors.step_curve,
                alpha=0.8,
                lw=1.5,
            )
            self.style_axis(
                ax, "Step Sizes", xlabel="Iteration", ylabel=r"$||x_{t+1} - x_t||$"
            )

            # 4. Exploration Efficiency
            ax = axs[1, 1]

            # Left Axis: Physical Movement
            ln1 = ax.plot(
                steps,
                data.cumulative_path,
                color=self.config.colors.path_curve,
                alpha=0.7,
                lw=1.5,
                label="Total Path",
            )
            ln2 = ax.plot(
                steps,
                data.displacement,
                color=self.config.colors.displacement_curve,
                alpha=0.9,
                lw=2.0,
                label="Displacement",
            )
            ax.fill_between(
                steps,
                data.displacement,
                data.cumulative_path,
                color=self.config.colors.path_curve,
                alpha=0.1,
                label="Redundant Motion",
            )
            self.style_axis(
                ax,
                "Exploration Efficiency",
                xlabel="Iteration",
                ylabel="Cumulative Distance",
            )

            # Right Axis: Efficiency Score
            ax2 = ax.twinx()
            score_label = "Eff. Score (0-1)"
            ln3 = ax2.plot(
                steps,
                data.efficiency_scores,
                color=self.config.colors.efficiency_fill,
                lw=1.5,
                ls="--",
                label=score_label,
            )
            ax2.fill_between(
                steps,
                0,
                data.efficiency_scores,
                color=self.config.colors.efficiency_fill,
                alpha=0.1,
            )

            ax2.set_ylim(-0.05, 1.05)
            ax2.set_ylabel(
                score_label,
                color=self.config.colors.efficiency_fill,
                fontsize=self.config.fonts.label_size,
                fontweight="bold",
            )
            ax2.tick_params(
                axis="y",
                labelcolor=self.config.colors.efficiency_fill,
                labelsize=self.config.fonts.tick_size,
            )
            ax2.spines["right"].set_visible(True)
            ax2.spines["right"].set_color(self.config.colors.efficiency_fill)
            ax2.spines["top"].set_visible(False)

            # Unified Legend
            lines = ln1 + ln2 + ln3
            labels = [l.get_label() for l in lines]
            ax.legend(
                lines,
                labels,
                loc="lower right",
                fontsize=self.config.fonts.legend_size,
                framealpha=0.9,
                edgecolor=self.config.colors.grid,
            )

    def plot_phase_portrait(self, filename: str, data: TrajectoryData, func_name: str):
        """
        Plots Step Size vs Gradient Norm.
        Helps diagnose if optimizer is unstable (high step, low grad) or stagnating (low step, high grad).
        """
        path = self.output_dir / filename

        # Prepare Data
        x_raw, y_raw = data.gradient_norms[:-1], data.step_sizes[:-1]
        iters = np.arange(len(x_raw))

        # Filter practically zero values to avoid log-scale issues
        mask = (x_raw > 1e-14) & (y_raw > 1e-14)
        x_cl, y_cl, iters_cl = x_raw[mask], y_raw[mask], iters[mask]

        # Compute Smoothed Data
        alpha_val = self.config.params.ema_smoothing_factor
        x_sm = compute_ema(x_cl, alpha_val)
        y_sm = compute_ema(y_cl, alpha_val)

        # Determine shared limits
        if len(x_cl) > 0:
            all_v = np.concatenate([x_cl, y_cl, x_sm, y_sm])
            l_min, l_max = all_v.min() * 0.5, all_v.max() * 2.0
        else:
            l_min, l_max = 1e-5, 1.0

        def draw_phase_subplot(ax, xd, yd, title, smooth=False):
            # Diagonal reference line (Ratio = 1)
            ax.plot(
                [l_min, l_max],
                [l_min, l_max],
                color=self.config.colors.phase_background,
                ls="--",
                lw=1,
                zorder=0,
            )

            lc_obj = None
            if len(xd) > 2:
                # Color code lines by iteration
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
                self.add_trajectory_markers(ax, xd, yd)
            else:
                ax.scatter(xd, yd, alpha=0.5)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(l_min, l_max)
            ax.set_ylim(l_min, l_max)
            ax.set_aspect("equal", adjustable="box")
            self.style_axis(
                ax, title, "Gradient Norm (Log)", "Step Size (Log)", log_grid=True
            )

            # Annotations for zones
            bbox_style = dict(
                boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#eeeeee"
            )
            ax.text(
                l_max * 0.5,
                l_min * 2,
                "STAGNATION",
                ha="right",
                va="bottom",
                color=self.config.colors.gradient_curve,
                bbox=bbox_style,
                **self.config.fonts.annotation,
            )
            ax.text(
                l_min * 2,
                l_max * 0.5,
                "INSTABILITY",
                ha="left",
                va="top",
                color=self.config.colors.displacement_curve,
                bbox=bbox_style,
                **self.config.fonts.annotation,
            )
            return lc_obj

        # Create Layout
        fig = plt.figure(figsize=self.config.sizes.phase)
        gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.05], wspace=0.15, figure=fig)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])

        try:
            draw_phase_subplot(
                ax_left, x_cl, y_cl, f"Raw Dynamics (Jitter) | {func_name}", False
            )
            lc = draw_phase_subplot(
                ax_right, x_sm, y_sm, f"Smoothed Trend (Flow) | {func_name}", True
            )

            legend_handles = [
                mpatches.Patch(color="purple", label="Trajectory"),
                plt.Line2D(
                    [0],
                    [0],
                    color=self.config.colors.phase_background,
                    lw=1,
                    ls="--",
                    label="Ratio = 1.0",
                ),
            ]
            ax_right.legend(
                handles=legend_handles,
                loc="upper right",
                fontsize=self.config.fonts.legend_size,
            )

            if lc:
                cb = fig.colorbar(lc, cax=cax)
                cb.set_label(
                    "Iteration",
                    rotation=270,
                    labelpad=15,
                    size=self.config.fonts.label_size,
                )
                cb.ax.tick_params(labelsize=self.config.fonts.tick_size)

            fig.savefig(path, bbox_inches="tight", dpi=self.config.dpi)
        finally:
            plt.close(fig)

    def plot_update_ratio(self, filename: str, data: TrajectoryData, func_name: str):
        """Plots the ratio between Step Size and Gradient Norm."""
        path = self.output_dir / filename

        with FigureContext(path, self.config.sizes.wide, dpi=self.config.dpi) as (
            fig,
            ax,
        ):
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = data.step_sizes / (data.gradient_norms + 1e-10)

            ax.plot(ratio, color=self.config.colors.ratio_curve, lw=1)
            ax.set_yscale("log")
            self.style_axis(
                ax,
                f"Update Ratio (Step/Grad) | {func_name}",
                "Iteration",
                "Ratio (Log Scale)",
                log_grid=True,
            )

            ax.axhline(1.0, c=self.config.colors.text_secondary, ls="--", alpha=0.5)
            ax.text(
                len(ratio) * 0.02,
                1.2,
                "Aggressive (> 1.0)",
                color=self.config.colors.text_secondary,
                **self.config.fonts.annotation,
            )

    def plot_penalty_distribution(self, filename: str, metrics: Dict, func_name: str):
        """Plots a donut chart of the Optuna tuning penalties."""
        path = self.output_dir / filename

        with FigureContext(path, self.config.sizes.donut, dpi=self.config.dpi) as (
            fig,
            ax,
        ):
            valid = sorted(
                [(k, float(v)) for k, v in metrics.items() if v > 0],
                key=lambda x: x[1],
                reverse=True,
            )

            if not valid:
                ax.text(
                    0.5,
                    0.5,
                    "No tuning metrics",
                    ha="center",
                    fontsize=self.config.fonts.label_size,
                )
                ax.axis("off")
                return

            keys, vals = zip(*valid)
            vals = np.array(vals)

            is_skewed = vals.max() / (vals.min() + 1e-9) > 500
            disp_vals = np.log1p(vals) if is_skewed else vals

            wedges, _ = ax.pie(
                disp_vals,
                startangle=90,
                colors=[self.config.colors.get_metric_color(k) for k in keys],
                wedgeprops=dict(width=0.4, edgecolor="white"),
            )

            ax.text(
                0,
                0,
                "Tuning\nObjectives",
                ha="center",
                va="center",
                color=self.config.colors.text_primary,
                **self.config.fonts.axis,
            )
            ax.set_title(
                f"Tuning Penalty Distribution | {func_name}",
                color=self.config.colors.text_primary,
                **self.config.fonts.title,
            )

            legend_labels = [
                f"{k.replace('_', ' ').title()}: {v:.4g} ({v / vals.sum() * 100:.1f}%)"
                for k, v in zip(keys, vals)
            ]
            ax.legend(
                wedges,
                legend_labels,
                title="Metrics",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=self.config.fonts.legend_size,
            )

            if is_skewed:
                plt.figtext(
                    0.5,
                    0.02,
                    "Log-scaled for visibility",
                    ha="center",
                    color=self.config.colors.text_secondary,
                    **self.config.fonts.annotation,
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
    Main function to generate visualization assets for a benchmark run.

    Args:
        func: The objective function being minimized.
        func_name: Name of the function.
        cords: Tensor of shape (2, N) containing trajectory coordinates.
        output_dir: Directory to save images.
        optimizer_name: Name of the optimizer.
        optimizer_params: Dictionary of best hyperparameters found.
        eval_metrics: Dictionary of evaluation scores (e.g., final distance).
        tune_metrics: Dictionary of penalty scores from hyperparameter tuning.
        error_rate: Final calculated error/score.
        global_minimums: Tensor of global minimum coordinates.
        eval_size: Tuple of (x_range, y_range) defining the plot bounds.
        res: Resolution for surface plot ("auto" or int).
        img_format: Output image format (e.g., "png", "jpg").
        debug: Enable debug print statements.

    Returns:
        Dictionary mapping plot types to their file paths.
    """
    if debug:
        print(
            f"[Vis] Generating visualization for {func_name} with {optimizer_name}..."
        )

    func_dir = Path(output_dir)
    func_dir.mkdir(parents=True, exist_ok=True)

    pts_np = cords.t().detach().cpu().numpy()
    gm_np = global_minimums.detach().cpu().numpy()

    vis = OptimizerVisualizer(func_dir, debug=debug)

    traj_data = vis.compute_metrics(func, pts_np)

    scaled_bounds = scale_eval_size(eval_size, vis.config.params.surface_padding_factor)
    X_tens, Y_tens, Z_tens = compute_surface(
        func, func_name, scaled_bounds, res, debug=debug
    )
    X, Y, Z = X_tens.numpy(), Y_tens.numpy(), Z_tens.numpy()

    param_str = ", ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in optimizer_params.items()
    )
    main_title = (
        f"{func_name} Landscape - {optimizer_name}\n"
        f"Iters: {len(pts_np) - 1}, Error: {error_rate:.4f}\n{param_str}"
    )

    files = {}

    f_surf = f"surface.{img_format}"
    vis.plot_surface(
        f_surf, X, Y, Z, traj_data, scaled_bounds, gm_np, main_title, eval_metrics
    )
    files["surface"] = str(func_dir / f_surf)

    f_dyn = f"dynamics.{img_format}"
    vis.plot_dynamics(f_dyn, traj_data, func_name)
    files["dynamics"] = str(func_dir / f_dyn)

    f_phase = f"phase_portrait.{img_format}"
    vis.plot_phase_portrait(f_phase, traj_data, func_name)
    files["phase_portrait"] = str(func_dir / f_phase)

    f_ratio = f"update_ratio.{img_format}"
    vis.plot_update_ratio(f_ratio, traj_data, func_name)
    files["update_ratio"] = str(func_dir / f_ratio)

    f_donut = f"penalty_donut.{img_format}"
    vis.plot_penalty_distribution(f_donut, tune_metrics, func_name)
    files["penalty_donut"] = str(func_dir / f_donut)

    return files
