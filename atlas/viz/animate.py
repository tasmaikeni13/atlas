"""Animated GIF rendering of loss landscape evolution and optimization trajectories."""

from __future__ import annotations

import os
import pathlib
import tempfile
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .style import set_publication_style, COLOR_MAP, TRAJECTORY_COLOR, MIN_COLOR

def create_landscape_gif(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    trajectory_2d: np.ndarray,
    output_path: str,
    fps: int = 6,
    title_prefix: str = "Optimization Dynamics"
) -> str:
    """Renders a smooth animated GIF showing the optimizer navigating the loss basin.
    
    Generates temporal frames tracking the trajectory progression across epochs.
    """
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    T = len(trajectory_2d)
    assert T >= 2, "Trajectory must contain at least 2 points for animation"

    min_val = np.min(Z)
    max_val = np.max(Z)
    levels = np.linspace(min_val, max_val, 32)

    frames = []
    # Sample up to 24 frames
    num_frames = min(T, 24)
    step_indices = np.linspace(1, T, num_frames, dtype=int)

    with tempfile.TemporaryDirectory() as tmpdir:
        for frame_idx, step_limit in enumerate(step_indices):
            fig, ax = plt.subplots(figsize=(7, 6))

            # Base contour
            cf = ax.contourf(X, Y, Z, levels=levels, cmap=COLOR_MAP, alpha=0.92)
            ax.contour(X, Y, Z, levels=10, colors="white", alpha=0.2, linewidths=0.5)

            # Sub-trajectory up to current step
            current_traj = trajectory_2d[:step_limit]
            ax.plot(current_traj[:, 0], current_traj[:, 1], color=TRAJECTORY_COLOR, linewidth=2.2, alpha=0.95, zorder=5)

            # Start point
            ax.scatter([trajectory_2d[0, 0]], [trajectory_2d[0, 1]], c="#FF5964", s=70, marker="X", edgecolors="white", zorder=6)
            
            # Current head
            ax.scatter([current_traj[-1, 0]], [current_traj[-1, 1]], c=MIN_COLOR, s=80, marker="o", edgecolors="white", linewidths=1.2, zorder=7)

            progress_pct = int((step_limit / T) * 100)
            ax.set_title(f"{title_prefix} — Progress: {progress_pct}% (Step {step_limit}/{T})", weight="bold", pad=10)
            ax.set_xlabel("$u$ (1st Subspace Component)", weight="bold")
            ax.set_ylabel("$v$ (2nd Subspace Component)", weight="bold")
            ax.set_xlim(np.min(X), np.max(X))
            ax.set_ylim(np.min(Y), np.max(Y))

            plt.tight_layout()
            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:03d}.png")
            fig.savefig(frame_path, dpi=160)
            plt.close(fig)

            img = Image.open(frame_path).convert("RGB")
            frames.append(img)

        # Save GIF via PIL
        duration_ms = int(1000 / fps)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True
        )

    return output_path
