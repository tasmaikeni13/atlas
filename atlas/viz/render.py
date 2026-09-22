"""Publication-grade rendering for loss landscapes, error certificates, and diagnostics."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from .style import (
    set_publication_style,
    COLOR_MAP,
    TRAJECTORY_COLOR,
    ANCHOR_COLOR,
    CERT_COLOR,
    MIN_COLOR,
)
from ..certify import Certificate
from ..reconstruct import HermiteTaylorReconstruction, SurfaceAnalysis

def render_landscape_2d(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    trajectory_2d: Optional[np.ndarray] = None,
    anchors: Optional[np.ndarray] = None,
    cert_points: Optional[np.ndarray] = None,
    analysis: Optional[SurfaceAnalysis] = None,
    certificate: Optional[Certificate] = None,
    title: str = "Certified Loss Landscape (ATLAS)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Renders a publication-quality 2D contour map with trajectory and certificates."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Log-scale contour levels if wide dynamic range
    min_val = np.min(Z)
    max_val = np.max(Z)
    
    # 2D Filled Contours
    levels = np.linspace(min_val, max_val, 35)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=COLOR_MAP, extend="both", alpha=0.92)
    cs = ax.contour(X, Y, Z, levels=12, colors="white", alpha=0.25, linewidths=0.6)
    
    cbar = fig.colorbar(cf, ax=ax, shrink=0.88, pad=0.04)
    cbar.set_label("Validation Loss", rotation=270, labelpad=16, weight="bold")

    # Plot Estimation Anchors
    if anchors is not None and len(anchors) > 0:
        ax.scatter(
            anchors[:, 0], anchors[:, 1],
            c=ANCHOR_COLOR, s=28, marker="o", edgecolors="black", linewidths=0.5,
            label=f"Taylor Anchors (N={len(anchors)})", zorder=4
        )

    # Plot Certification Holdout Points
    if cert_points is not None and len(cert_points) > 0:
        ax.scatter(
            cert_points[:, 0], cert_points[:, 1],
            c=CERT_COLOR, s=32, marker="^", edgecolors="black", linewidths=0.5,
            label=f"Cert Holdouts (M={len(cert_points)})", zorder=4
        )

    # Plot Optimizer Trajectory
    if trajectory_2d is not None and len(trajectory_2d) > 1:
        ax.plot(
            trajectory_2d[:, 0], trajectory_2d[:, 1],
            color=TRAJECTORY_COLOR, linewidth=2.0, alpha=0.95, zorder=5,
            label="Optimization Trajectory"
        )
        # Start and End points
        ax.scatter(
            [trajectory_2d[0, 0]], [trajectory_2d[0, 1]],
            c="#FF5964", s=70, marker="X", edgecolors="white", linewidths=1.0,
            label="Initialization", zorder=6
        )
        ax.scatter(
            [trajectory_2d[-1, 0]], [trajectory_2d[-1, 1]],
            c=MIN_COLOR, s=70, marker="*", edgecolors="white", linewidths=1.0,
            label="Converged Solution", zorder=6
        )

    ax.set_xlabel("Subspace Basis Vector $u$ (Top Trajectory PC)", weight="bold")
    ax.set_ylabel("Subspace Basis Vector $v$ (2nd Trajectory PC)", weight="bold")
    ax.set_title(title, weight="bold", pad=12)

    # Add error certificate annotation badge
    if certificate is not None:
        badge_text = (
            f"Certified Accuracy (95% coverage):\n"
            f"Error bound: +/-{certificate.q95_upper_bound:.4f} ({certificate.relative_q95_error_pct:.1f}% relief)\n"
            f"Confidence: {certificate.confidence_level*100:.0f}% | Holdouts: M={certificate.num_cert_points}"
        )
        ax.text(
            0.03, 0.04, badge_text,
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.75, edgecolor="#00F5D4"),
            color="white", zorder=7
        )

    ax.legend(loc="upper right", framealpha=0.85, facecolor="white", edgecolor="none")
    ax.set_xlim(np.min(X), np.max(X))
    ax.set_ylim(np.min(Y), np.max(Y))

    plt.tight_layout()
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        # Also save PDF if PNG requested
        if save_path.endswith(".png"):
            pdf_path = save_path[:-4] + ".pdf"
            fig.savefig(pdf_path)
    return fig


def render_landscape_3d(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    trajectory_2d: Optional[np.ndarray] = None,
    title: str = "3D Certified Loss Basin (ATLAS)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Renders a 3D perspective wireframe-surface view of the loss landscape."""
    set_publication_style()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Surface render
    surf = ax.plot_surface(
        X, Y, Z, cmap=COLOR_MAP, linewidth=0.2, antialiased=True, alpha=0.88,
        rstride=2, cstride=2, edgecolor="black"
    )
    
    # Trajectory overlay on 3D surface
    if trajectory_2d is not None and len(trajectory_2d) > 1:
        # Interpolate Z values along trajectory
        from scipy.interpolate import RegularGridInterpolator
        xs = X[0, :]
        ys = Y[:, 0]
        interp = RegularGridInterpolator((ys, xs), Z, bounds_error=False, fill_value=None)
        traj_z = interp(np.column_stack([trajectory_2d[:, 1], trajectory_2d[:, 0]]))
        ax.plot(
            trajectory_2d[:, 0], trajectory_2d[:, 1], traj_z + 0.05 * (np.max(Z) - np.min(Z)),
            color=TRAJECTORY_COLOR, linewidth=2.5, zorder=10, label="Optimizer Trajectory"
        )

    ax.view_init(elev=35, azim=225)
    ax.set_xlabel("$u$", labelpad=8)
    ax.set_ylabel("$v$", labelpad=8)
    ax.set_zlabel("Loss", labelpad=8)
    ax.set_title(title, weight="bold", pad=10)

    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Loss")
    plt.tight_layout()

    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        if save_path.endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf")
    return fig


def render_certificate_plot(
    residuals: np.ndarray,
    certificate: Certificate,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Renders empirical cumulative error distribution and finite-sample DKW bounds."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    sorted_res = np.sort(residuals)
    N = len(sorted_res)
    ecdf = np.arange(1, N + 1) / N

    # Plot empirical CDF
    ax.step(sorted_res, ecdf, where="post", color="#3A86FF", linewidth=2.0, label="Empirical Error CDF $\\widehat{F}_M(e)$")

    # DKW Confidence band
    alpha = 1.0 - certificate.confidence_level
    eps_dkw = np.sqrt(np.log(2.0 / alpha) / (2.0 * N))
    ecdf_lower = np.maximum(ecdf - eps_dkw, 0.0)
    ecdf_upper = np.minimum(ecdf + eps_dkw, 1.0)
    ax.fill_between(sorted_res, ecdf_lower, ecdf_upper, color="#3A86FF", alpha=0.2, label=f"DKW Band ({certificate.confidence_level*100:.0f}% Conf)")

    # 95% threshold line
    ax.axhline(0.95, color="#EF476F", linestyle="--", linewidth=1.2, label="95% Coverage Target")
    ax.axvline(certificate.q95_upper_bound, color="#06D6A0", linestyle="-.", linewidth=1.5,
               label=f"Cert Bound $\\epsilon_{{95}} = {certificate.q95_upper_bound:.4f}$")

    ax.set_xlabel("Reconstruction Error Residual $|\\widehat{L}(z) - L_{{\\mathrm{{true}}}}(z)|$", weight="bold")
    ax.set_ylabel("Cumulative Probability", weight="bold")
    ax.set_title(f"ATLAS Statistical Error Certificate (Holdout M={N})", weight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        if save_path.endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf")
    return fig


def render_sharpness_comparison_plot(
    batch_sizes: List[int],
    grid_sharpness: List[float],
    true_sharpness: float,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Visualizes sharpness inflation on mini-batch finite-difference grids vs exact Hessian."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(batch_sizes, grid_sharpness, "o-", color="#EF476F", linewidth=2.0, markersize=6, label="Finite-Difference Grid (Li et al.)")
    ax.axhline(true_sharpness, color="#06D6A0", linewidth=2.2, linestyle="--", label=f"ATLAS Exact Hessian Jet: $\\lambda_1 = {true_sharpness:.2f}$")

    # Annotate inflation
    inflation_small = grid_sharpness[0] / max(true_sharpness, 1e-4)
    ax.annotate(
        f"{inflation_small:.1f}x Inflation (Noise $h^{{-2}}$)",
        xy=(batch_sizes[0], grid_sharpness[0]),
        xytext=(batch_sizes[0] * 1.5, grid_sharpness[0] * 0.85),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
        fontsize=9, weight="bold"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Evaluation Mini-Batch Size $B$", weight="bold")
    ax.set_ylabel("Estimated Sharpness (Curvature)", weight="bold")
    ax.set_title("Sharpness Overstatement: Grid Finite-Difference vs Exact Jet", weight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        if save_path.endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf")
    return fig


def render_benchmark_rate_plot(
    budgets: List[float],
    atlas_errors: List[float],
    grid_errors: List[float],
    random_errors: List[float],
    taylor_errors: List[float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Log-log convergence rates comparing ATLAS against competing methods."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.plot(budgets, grid_errors, "s--", color="#FFB703", linewidth=1.8, label="Uniform 2D Grid (Li et al. 2018)")
    ax.plot(budgets, random_errors, "d--", color="#FB8500", linewidth=1.8, label="Random 2D Slice Grid")
    ax.plot(budgets, taylor_errors, "^:", color="#8E9AAF", linewidth=1.8, label="Global Taylor (1 Anchor)")
    ax.plot(budgets, atlas_errors, "o-", color="#06D6A0", linewidth=2.5, markersize=7, label="ATLAS (Adaptive Partitioned Jets)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget $C$ (Normalized Seconds / FLOPs)", weight="bold")
    ax.set_ylabel("Surface Reconstruction Error $L_2(\\Omega)$", weight="bold")
    ax.set_title("Reconstruction Error vs Compute Budget: Empirical Minimax Scaling", weight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        if save_path.endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf")
    return fig
