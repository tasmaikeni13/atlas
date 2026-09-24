"""Budget-optimal experimental design and anchor allocation for ATLAS."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .probe import CostModel

@dataclasses.dataclass
class OptimalAllocation:
    """Feasible integer allocation minimizing the configured error surrogate."""
    n_total: int
    n_est: int
    n_cert: int
    batch_size: int
    budget_seconds: float
    expected_error: float
    c_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_total": int(self.n_total),
            "n_est": int(self.n_est),
            "n_cert": int(self.n_cert),
            "batch_size": int(self.batch_size),
            "budget_seconds": float(self.budget_seconds),
            "expected_error": float(self.expected_error),
            "c_ratio": float(self.c_ratio),
        }


class BudgetAllocator:
    """Minimize the stated error surrogate over feasible integer allocations."""

    def __init__(self, cost_model: CostModel, cert_fraction: float = 0.15):
        self.cm = cost_model
        self.cert_fraction = cert_fraction

    def solve(
        self,
        budget_seconds: float,
        radius: float = 1.0,
        min_batch: int = 16,
        max_batch: int = 2048,
        min_anchors: int = 8,
        max_anchors: int = 80
    ) -> OptimalAllocation:
        """Minimize the error surrogate subject to the affine cost budget.

        The estimation count, rather than the total count including holdouts,
        controls the spatial term. A constrained finite search is exact for
        this surrogate because error decreases with batch size at fixed N.
        """
        tau = self.cm.tau
        kappa = self.cm.kappa
        if tau < 0 or kappa <= 0:
            raise ValueError("Cost model requires tau >= 0 and kappa > 0")
        if not 0.0 <= self.cert_fraction < 1.0:
            raise ValueError("cert_fraction must be in [0, 1)")
        if not 0 < min_batch <= max_batch or not 0 < min_anchors <= max_anchors:
            raise ValueError("Invalid allocation limits")
        sigma = np.sqrt(max(self.cm.sigma2, 1e-8))
        m3 = max(self.cm.m3, 1e-4)

        c1 = 1.0 / 6.0
        c2 = 1.0

        best = None
        for n_total in range(min_anchors, max_anchors + 1):
            n_cert = max(int(round(n_total * self.cert_fraction)), 4)
            n_est = n_total - n_cert
            if n_est < 4:
                continue
            affordable = (budget_seconds / n_total - tau) / kappa
            batch_size = min(max_batch, int(np.floor(affordable)))
            while batch_size >= min_batch and n_total * (tau + kappa * batch_size) > budget_seconds:
                batch_size -= 1
            if batch_size < min_batch:
                continue
            cost = n_total * (tau + kappa * batch_size)
            error = float(
                c1 * m3 * radius ** 3 * n_est ** (-1.5)
                + c2 * sigma / np.sqrt(batch_size)
            )
            candidate = (error, cost, n_total, n_est, n_cert, batch_size)
            if best is None or candidate < best:
                best = candidate

        if best is None:
            raise ValueError("No allocation satisfies the wall-clock budget")
        exp_err, _, best_N, n_est, n_cert, best_B = best
        cost_per_anchor = tau + kappa * best_B

        return OptimalAllocation(
            n_total=best_N,
            n_est=n_est,
            n_cert=n_cert,
            batch_size=best_B,
            budget_seconds=budget_seconds,
            expected_error=float(exp_err),
            c_ratio=float((kappa * best_B) / cost_per_anchor)
        )


def generate_halton_anchors(
    num_points: int,
    radius_x: float = 1.0,
    radius_y: float = 1.0,
    seed: int = 42
) -> np.ndarray:
    """Generates 2D quasi-random low-discrepancy anchor points in [-rx, rx] x [-ry, ry]."""
    # Halton sequence for bases (2, 3)
    def halton_seq(count: int, base: int) -> np.ndarray:
        seq = np.zeros(count)
        for i in range(count):
            f = 1.0
            r = 0.0
            idx = i + 1 + seed
            while idx > 0:
                f /= base
                r += f * (idx % base)
                idx //= base
            seq[i] = r
        return seq

    u = halton_seq(num_points, 2)
    v = halton_seq(num_points, 3)

    # Scale to [-rx, rx] x [-ry, ry]
    xs = (u * 2.0 - 1.0) * radius_x
    ys = (v * 2.0 - 1.0) * radius_y
    anchors = np.column_stack([xs, ys])
    # Ensure origin (0, 0) is explicitly included
    anchors[0] = [0.0, 0.0]
    return anchors


def generate_dense_grid(
    radius_x: float = 1.0,
    radius_y: float = 1.0,
    resolution: int = 80
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates high-resolution coordinate matrices (X, Y) and flattened queries for visualization."""
    xs = np.linspace(-radius_x, radius_x, resolution, dtype=np.float32)
    ys = np.linspace(-radius_y, radius_y, resolution, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    query_points = np.column_stack([X.ravel(), Y.ravel()])
    return X, Y, query_points
