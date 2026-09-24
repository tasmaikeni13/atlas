"""Hermite-Taylor Partition of Unity loss surface reconstruction."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree

from .probe import Jet

@dataclasses.dataclass
class SurfaceAnalysis:
    """Analytical geometry extracted from the reconstructed loss landscape."""
    min_coord: Tuple[float, float]
    min_loss: float
    origin_loss: float
    origin_curvature_eigs: Tuple[float, float]
    origin_condition_number: float
    flatness_index: float
    surface_relief: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_coord": [float(c) for c in self.min_coord],
            "min_loss": float(self.min_loss),
            "origin_loss": float(self.origin_loss),
            "origin_curvature_eigs": [float(e) for e in self.origin_curvature_eigs],
            "origin_condition_number": float(self.origin_condition_number),
            "flatness_index": float(self.flatness_index),
            "surface_relief": float(self.surface_relief),
        }


class HermiteTaylorReconstruction:
    """Synthesizes a smooth loss surface from discrete 2nd-order Taylor jets.
    
    Combines local quadratic Taylor models via a partition of unity with smooth blending kernels.
    """

    def __init__(self, jets: Sequence[Jet], p_power: float = 3.0, eps: float = 1e-4):
        self.jets = list(jets)
        assert len(self.jets) >= 1, "Must provide at least 1 Jet for reconstruction"
        if p_power <= 0 or eps <= 0:
            raise ValueError("p_power and eps must be positive")
        self.anchor_coords = np.array([[j.x, j.y] for j in self.jets], dtype=np.float64)  # (N, 2)
        self.p_power = p_power
        self.eps = eps

        # Precompute k-nearest neighbor bandwidths for adaptive scale
        if len(self.jets) >= 4:
            tree = cKDTree(self.anchor_coords)
            k = min(4, len(self.jets))
            dists, _ = tree.query(self.anchor_coords, k=k)
            # Use distance to k-th neighbor as local radius
            self.radii = np.maximum(dists[:, -1], 1e-3)
        else:
            self.radii = np.ones(len(self.jets), dtype=np.float64)

    def evaluate_batch(self, query_points: np.ndarray) -> np.ndarray:
        """Evaluates the reconstructed loss surface at query points of shape (M, 2).
        
        Returns:
            losses: 1D array of shape (M,)
        """
        # query_points: (M, 2)
        M = query_points.shape[0]
        N = len(self.jets)

        # Distances from each query point to each anchor: (M, N)
        diff = query_points[:, None, :] - self.anchor_coords[None, :, :]  # (M, N, 2)
        dist_sq = np.sum(diff ** 2, axis=-1)  # (M, N)
        # Evaluate local Taylor polynomials P_k(q) for all (M, N)
        poly_vals = np.zeros((M, N), dtype=np.float64)
        for k, jet in enumerate(self.jets):
            dx = diff[:, k, 0]
            dy = diff[:, k, 1]
            lin = jet.grad[0] * dx + jet.grad[1] * dy
            quad = 0.5 * (jet.hess[0, 0] * (dx ** 2) + 2.0 * jet.hess[0, 1] * dx * dy + jet.hess[1, 1] * (dy ** 2))
            poly_vals[:, k] = jet.loss + lin + quad

        # Smooth inverse-distance Shepard weights (not compactly supported).
        scaled_dist_sq = dist_sq / (self.radii[None, :] ** 2)
        log_weights = -0.5 * self.p_power * np.log(scaled_dist_sq + self.eps ** 2)
        weights = np.exp(log_weights - np.max(log_weights, axis=1, keepdims=True))
        norm_weights = weights / np.sum(weights, axis=1, keepdims=True)

        pred_losses = np.sum(norm_weights * poly_vals, axis=1)
        return pred_losses

    def evaluate_at(self, x: float, y: float) -> float:
        """Evaluates loss at a single coordinate (x, y)."""
        pt = np.array([[x, y]], dtype=np.float64)
        return float(self.evaluate_batch(pt)[0])

    def query_curvature(self, x: float, y: float) -> float:
        """Evaluates maximum directional curvature at coordinate (x, y)."""
        pt = np.array([x, y], dtype=np.float64)
        dists = np.linalg.norm(self.anchor_coords - pt, axis=1)
        nearest_idx = int(np.argmin(dists))
        jet = self.jets[nearest_idx]
        eigs = np.linalg.eigvalsh(jet.hess)
        return float(np.max(eigs))

    def analyze_geometry(self, grid_X: np.ndarray, grid_Y: np.ndarray) -> SurfaceAnalysis:
        """Extracts key topological metrics (minimum, condition number, relief)."""
        queries = np.column_stack([grid_X.ravel(), grid_Y.ravel()])
        Z = self.evaluate_batch(queries).reshape(grid_X.shape)

        min_idx = np.unravel_index(np.argmin(Z), Z.shape)
        min_coord = (float(grid_X[min_idx]), float(grid_Y[min_idx]))
        min_loss = float(Z[min_idx])
        origin_loss = self.evaluate_at(0.0, 0.0)

        # Inspect origin anchor Hessian (anchor 0 is at origin 0,0)
        origin_jet = self.jets[0]
        eigs = np.linalg.eigvalsh(origin_jet.hess)
        lambda1, lambda2 = float(eigs[1]), float(eigs[0])
        cond = float(abs(lambda1) / max(abs(lambda2), 1e-6))
        flatness = float(lambda1 + lambda2)
        relief = float(np.max(Z) - np.min(Z))

        return SurfaceAnalysis(
            min_coord=min_coord,
            min_loss=min_loss,
            origin_loss=origin_loss,
            origin_curvature_eigs=(lambda1, lambda2),
            origin_condition_number=cond,
            flatness_index=flatness,
            surface_relief=relief
        )
