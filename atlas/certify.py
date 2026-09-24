"""Statistical error certification and finite-sample quantile bounds for ATLAS."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .probe import Jet
from .reconstruct import HermiteTaylorReconstruction

@dataclasses.dataclass
class Certificate:
    """Holdout errors and, when justified, a finite-sample DKW bound."""
    num_cert_points: int
    mae: float
    rmse: float
    max_error: float
    q95_error: float
    q95_upper_bound: float  # With finite-sample DKW correction
    confidence_level: float
    surface_relief: float
    relative_q95_error_pct: float
    certified_valid: bool
    coverage_lower_bound: float = 0.0
    sampling: str = "unspecified"
    target: str = "fixed evaluation-batch loss"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_cert_points": int(self.num_cert_points),
            "mae": float(self.mae),
            "rmse": float(self.rmse),
            "max_error": float(self.max_error),
            "q95_error": float(self.q95_error),
            "q95_upper_bound": float(self.q95_upper_bound),
            "confidence_level": float(self.confidence_level),
            "surface_relief": float(self.surface_relief),
            "relative_q95_error_pct": float(self.relative_q95_error_pct),
            "certified_valid": bool(self.certified_valid),
            "coverage_lower_bound": float(self.coverage_lower_bound),
            "sampling": self.sampling,
            "target": self.target,
        }

    def summary(self) -> str:
        if not self.certified_valid:
            return (
                f"Empirical holdout q95={self.q95_error:.4g}, max={self.max_error:.4g} "
                f"(M={self.num_cert_points}); 95% domain coverage is not certified. "
                f"Sampling: {self.sampling}."
            )
        return (
            f"Certified: 95% of fixed-batch domain within +/-{self.q95_upper_bound:.4g} "
            f"({self.relative_q95_error_pct:.2f}% of loss relief) at {self.confidence_level * 100:.0f}% confidence "
            f"(M={self.num_cert_points} hold-out points, RMSE={self.rmse:.4g})"
        )


def certify_reconstruction(
    reconstruction: HermiteTaylorReconstruction,
    cert_jets: Sequence[Jet],
    surface_relief: float,
    confidence_level: float = 0.95,
    *,
    iid_uniform_coords: bool = False,
) -> Certificate:
    """Evaluate held-out losses on a fixed evaluation batch.

    A domain-wide DKW claim requires coordinates sampled independently and
    uniformly *after* the reconstruction is fixed. With fewer points than the
    DKW rank requires, the maximum observed residual is only descriptive.
    This says nothing about the population loss unless that loss is evaluated
    exactly or an additional observation-error bound is supplied.
    """
    if len(cert_jets) < 2:
        raise ValueError("Need at least two holdout points")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    M = len(cert_jets)

    query_pts = np.array([[j.x, j.y] for j in cert_jets], dtype=np.float64)
    true_losses = np.array([j.loss for j in cert_jets], dtype=np.float64)
    pred_losses = reconstruction.evaluate_batch(query_pts)

    residuals = np.abs(pred_losses - true_losses)

    mae = float(np.mean(residuals))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    max_err = float(np.max(residuals))

    # Empirical 95th percentile
    q95 = float(np.percentile(residuals, 95))

    # DKW finite-sample confidence band correction:
    # epsilon_dkw = sqrt(ln(2 / alpha) / (2M))
    alpha = 1.0 - confidence_level
    eps_dkw = np.sqrt(np.log(2.0 / alpha) / (2.0 * M))
    
    # Adjusted rank index ensuring (1 - alpha) statistical coverage
    target_quantile = 0.95 + eps_dkw
    certified_valid = bool(iid_uniform_coords and target_quantile <= 1.0)
    if certified_valid:
        # Inverse ECDF is an order statistic. Interpolated percentiles can
        # understate the finite-sample upper bound.
        rank = math.ceil(M * target_quantile) - 1
        q95_upper = float(np.sort(residuals)[rank])
    else:
        q95_upper = max_err

    safe_relief = max(surface_relief, 1e-4)
    rel_pct = float((q95_upper / safe_relief) * 100.0)

    return Certificate(
        num_cert_points=M,
        mae=mae,
        rmse=rmse,
        max_error=max_err,
        q95_error=q95,
        q95_upper_bound=q95_upper,
        confidence_level=confidence_level,
        surface_relief=surface_relief,
        relative_q95_error_pct=rel_pct,
        certified_valid=certified_valid,
        coverage_lower_bound=max(0.0, 1.0 - eps_dkw) if iid_uniform_coords else 0.0,
        sampling="iid uniform" if iid_uniform_coords else "unverified or deterministic",
    )
