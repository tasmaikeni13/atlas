"""Industrial loss landscape diagnostics for Transformer optimization and training monitoring."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from .probe import Jet

@dataclasses.dataclass
class SharpnessComparison:
    """Comparison between true analytical Hessian sharpness and finite-difference grid estimates."""
    true_hessian_sharpness: float
    grid_finite_diff_sharpness: float
    inflation_factor: float
    noise_variance_floor: float
    discretization_step: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "true_hessian_sharpness": float(self.true_hessian_sharpness),
            "grid_finite_diff_sharpness": float(self.grid_finite_diff_sharpness),
            "inflation_factor": float(self.inflation_factor),
            "noise_variance_floor": float(self.noise_variance_floor),
            "discretization_step": float(self.discretization_step),
        }


@dataclasses.dataclass
class TrajectoryDiagnostic:
    """Diagnostic telemetry captured across optimizer training checkpoints."""
    checkpoint_steps: List[int]
    train_losses: List[float]
    val_losses: List[float]
    curvature_trace: List[float]
    condition_numbers: List[float]
    subspace_capture_ratios: List[float]
    effective_ranks: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_steps": [int(s) for s in self.checkpoint_steps],
            "train_losses": [float(l) for l in self.train_losses],
            "val_losses": [float(l) for l in self.val_losses],
            "curvature_trace": [float(c) for c in self.curvature_trace],
            "condition_numbers": [float(k) for k in self.condition_numbers],
            "subspace_capture_ratios": [float(r) for r in self.subspace_capture_ratios],
            "effective_ranks": [float(e) for e in self.effective_ranks] if self.effective_ranks else None,
        }


class LandscapeDiagnostics:
    """Diagnostic suite for industrial model inspection, stability analysis, and basin profiling."""

    @staticmethod
    def compare_sharpness(
        jet: Jet,
        eval_loss_fn: callable,
        coords: Tuple[float, float] = (0.0, 0.0),
        h: float = 0.05,
        batch_sample: Any = None
    ) -> SharpnessComparison:
        """Contrasts exact analytical Hessian sharpness against finite-difference grid stencil.
        
        Demonstrates the noise explosion (sigma / h^2) inherent to mini-batched grid evaluation.
        """
        # True analytical curvature: maximum eigenvalue of 2x2 Hessian
        eigs = np.linalg.eigvalsh(jet.hess)
        true_sharpness = float(np.max(eigs))

        x, y = coords
        # Finite-difference stencil along maximum curvature eigenvector v_max
        idx_max = np.argmax(eigs)
        _, eigvecs = np.linalg.eigh(jet.hess)
        v_max = eigvecs[:, idx_max]

        dx = float(h * v_max[0])
        dy = float(h * v_max[1])

        l_center = eval_loss_fn(x, y, batch_sample)
        l_fwd = eval_loss_fn(x + dx, y + dy, batch_sample)
        l_bwd = eval_loss_fn(x - dx, y - dy, batch_sample)

        fd_sharpness = float((l_fwd - 2.0 * l_center + l_bwd) / (h ** 2))
        inflation = float(fd_sharpness / max(true_sharpness, 1e-6))
        noise_floor = float(jet.variance / (h ** 4 + 1e-12)) if jet.variance > 0 else 0.0

        return SharpnessComparison(
            true_hessian_sharpness=true_sharpness,
            grid_finite_diff_sharpness=fd_sharpness,
            inflation_factor=inflation,
            noise_variance_floor=noise_floor,
            discretization_step=h
        )

    @staticmethod
    def compute_effective_rank(activations: np.ndarray) -> float:
        """Computes Roy-Vetterli effective rank of representation matrix (N x D):
        
        erank = exp( - sum_k p_k ln p_k ), where p_k = sigma_k / sum_j sigma_j.
        Detects dimensional collapse in Transformer attention features.
        """
        assert activations.ndim == 2, "Activations must be 2D matrix (batch, features)"
        s = np.linalg.svd(activations, compute_uv=False)
        s_sum = np.sum(s)
        if s_sum < 1e-12:
            return 1.0
        p = s / s_sum
        p = p[p > 1e-12]
        entropy = -np.sum(p * np.log(p))
        return float(np.exp(entropy))
