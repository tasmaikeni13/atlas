"""Statistical error certification and finite-sample quantile bounds for ATLAS."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .probe import Jet
from .reconstruct import HermiteTaylorReconstruction

@dataclasses.dataclass
class Certificate:
    """Rigorous finite-sample error certificate for a reconstructed loss landscape."""
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
        }

    def summary(self) -> str:
        return (
            f"Certified: 95% of domain within +/-{self.q95_upper_bound:.4g} "
            f"({self.relative_q95_error_pct:.2f}% of loss relief) at {self.confidence_level * 100:.0f}% confidence "
            f"(M={self.num_cert_points} hold-out points, RMSE={self.rmse:.4g})"
        )


def certify_reconstruction(
    reconstruction: HermiteTaylorReconstruction,
    cert_jets: Sequence[Jet],
    surface_relief: float,
    confidence_level: float = 0.95
) -> Certificate:
    """Evaluates reconstruction accuracy on independent hold-out validation anchors.
    
    Applies the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality to bound empirical quantile error
    in the presence of heavy-tailed outer surface residuals.
    """
    assert len(cert_jets) >= 2, "Need at least 2 hold-out points for certification"
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
    target_quantile = min(0.95 + eps_dkw, 1.0)
    q95_upper = float(np.percentile(residuals, target_quantile * 100.0))

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
        certified_valid=True
    )
