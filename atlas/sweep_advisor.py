"""Loss Landscape Diagnostic Engine & Educated Hyperparameter Sweep Advisor on TPUs.

Extracts fast, cheap, and exact geometric diagnostics (curvature sharpness, edge-of-stability margin,
basin conditioning, and stochastic SNR) to guide automated hyperparameter sweeping.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from .probe import Jet, JetProbe, CostModel
from .basis import SubspaceBasis


def directional_step_scale(
    jet: Jet,
    projected_update: np.ndarray,
    capture_ratio: float,
    max_scale: float = 8.0,
) -> Optional[float]:
    """Propose a scale from the jet's line minimum within a trust cap.

    The returned factor scales the learning rate that produced the update.
    It is only a local proposal: future gradients and optimizer state change,
    and a 2D plane may miss important directions.
    """
    update = np.asarray(projected_update, dtype=np.float64)
    if update.shape != (2,) or not np.all(np.isfinite(update)):
        raise ValueError("projected_update must be a finite 2-vector")
    if not np.isfinite(max_scale) or max_scale < 1.0:
        raise ValueError("max_scale must be finite and at least one")
    if not np.isfinite(capture_ratio) or capture_ratio < 0.5:
        return None

    slope = float(np.asarray(jet.grad, dtype=np.float64) @ update)
    curvature = float(update @ np.asarray(jet.hessian, dtype=np.float64) @ update)
    if not np.isfinite(slope) or not np.isfinite(curvature):
        return None
    if slope >= 0.0 or curvature <= 1e-12:
        return None
    optimum = -slope / curvature
    moderated_scale = 1.0 + min(capture_ratio, 1.0) * (optimum - 1.0)
    return float(np.clip(moderated_scale, 0.25, max_scale))


@dataclasses.dataclass
class LandscapeDiagnostics:
    """Comprehensive loss landscape diagnostic profile at a training checkpoint."""
    loss: float
    grad_norm: float
    lambda_max: float
    lambda_min: float
    condition_number: float
    anisotropy_ratio: float
    flatness_radius: float       # Radius r where loss rises by delta_L = 0.1
    eos_margin: float            # 2.0 / (lr * lambda_max)
    stochastic_snr: float        # ||grad||^2 / (sigma^2 / B)
    stability_verdict: str       # OPTIMAL_EDGE, OSCILLATING_UNSTABLE, SLUGGISH_UNDERFIT, ILL_CONDITIONED
    recommended_lr: float
    recommended_weight_decay: float
    recommended_batch_scale: float
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss": float(self.loss),
            "grad_norm": float(self.grad_norm),
            "lambda_max": float(self.lambda_max),
            "lambda_min": float(self.lambda_min),
            "condition_number": float(self.condition_number),
            "anisotropy_ratio": float(self.anisotropy_ratio),
            "flatness_radius": float(self.flatness_radius),
            "eos_margin": float(self.eos_margin),
            "stochastic_snr": float(self.stochastic_snr),
            "stability_verdict": self.stability_verdict,
            "recommended_lr": float(self.recommended_lr),
            "recommended_weight_decay": float(self.recommended_weight_decay),
            "recommended_batch_scale": float(self.recommended_batch_scale),
            "summary": self.summary,
        }


class LandscapeDiagnosticEngine:
    """Computes projected loss diagnostics from ATLAS Taylor jets."""

    def __init__(
        self,
        probe: JetProbe,
        current_lr: float = 3e-4,
        current_wd: float = 0.01,
        batch_size: int = 64,
        min_lr: float = 1e-5,
        max_lr: float = 3e-3
    ):
        self.probe = probe
        self.current_lr = current_lr
        self.current_wd = current_wd
        self.batch_size = batch_size
        self.min_lr = min_lr
        self.max_lr = max_lr

    def analyze(
        self,
        batch: Any,
        cost_model: Optional[CostModel] = None,
        delta_l_threshold: float = 0.1
    ) -> LandscapeDiagnostics:
        """Evaluates exact 2D Taylor Jet and returns actionable hyperparameter diagnostics."""
        jet = self.probe.evaluate_jet(0.0, 0.0, batch)
        return self.analyze_jet(jet, cost_model, delta_l_threshold)

    def analyze_jet(
        self,
        jet: Jet,
        cost_model: Optional[CostModel] = None,
        delta_l_threshold: float = 0.1,
    ) -> LandscapeDiagnostics:
        """Analyze a previously computed jet without repeating the probe."""
        loss = float(jet.loss)
        grad_norm = float(np.linalg.norm(jet.grad))

        # Compute projected 2x2 Hessian eigenvalues
        eigs = np.linalg.eigvalsh(jet.hessian)
        lambda_min = float(eigs[0])
        lambda_max = float(eigs[1])

        # Condition number: ratio of principal curvatures
        cond = float(abs(lambda_max) / max(abs(lambda_min), 1e-6))
        anisotropy = float(abs(lambda_max - lambda_min) / max(abs(lambda_max + lambda_min), 1e-6))

        # Flatness radius: r = sqrt(2 * delta_L / lambda_max)
        if lambda_max > 1e-6:
            flatness_radius = float(math.sqrt(2.0 * delta_l_threshold / lambda_max))
        else:
            flatness_radius = 5.0  # Very flat basin

        # Edge of Stability (Cohen et al. 2021) margin: mu = 2.0 / (lr * lambda_max)
        if lambda_max > 1e-7:
            eos_margin = float(2.0 / (self.current_lr * lambda_max))
        else:
            eos_margin = 100.0

        # Stochastic Signal-to-Noise Ratio: ||grad||^2 / (sigma^2 / B)
        sigma2 = cost_model.sigma2 if cost_model is not None else 1e-3
        denom = max(sigma2 / max(self.batch_size, 1), 1e-8)
        snr = float((grad_norm ** 2) / denom)

        # Formulate Educated Diagnoses & Next Sweep Recommendations
        # 1. Stability Verdict
        if eos_margin < 0.9:
            verdict = "OSCILLATING_UNSTABLE"
            rec_lr = max(self.min_lr, min(self.max_lr, self.current_lr * 0.40))
            rec_wd = self.current_wd * 1.5
            rec_b_scale = 1.5
            summary = (
                f"🚨 DIVERGENCE WARNING: Operating past Edge of Stability (EoS margin = {eos_margin:.2f} < 1.0). "
                f"Trajectory is bouncing across canyon walls with lambda_max = {lambda_max:.2f}. "
                f"Recommend reducing LR from {self.current_lr:.1e} to {rec_lr:.1e}."
            )
        elif 0.9 <= eos_margin <= 3.0:
            verdict = "OPTIMAL_EDGE"
            rec_lr = max(self.min_lr, min(self.max_lr, self.current_lr))
            rec_wd = self.current_wd
            rec_b_scale = 1.0
            summary = (
                f"✅ OPTIMAL REGIME: Operating directly along Edge of Stability (EoS margin = {eos_margin:.2f}). "
                f"Learning rate {self.current_lr:.1e} maximizes training speed while staying stable. "
                f"Basin flatness radius = {flatness_radius:.3f}."
            )
        else:
            verdict = "SLUGGISH_UNDERFIT"
            rec_lr = max(self.min_lr, min(self.max_lr, self.current_lr * min(2.5, max(1.2, eos_margin / 2.0))))
            rec_wd = self.current_wd
            rec_b_scale = 1.0
            summary = (
                f"🐢 UNDERFIT WARNING: Learning rate is overly conservative (EoS margin = {eos_margin:.2f} >> 2.0). "
                f"Landscape can tolerate up to {rec_lr:.1e} with lambda_max = {lambda_max:.4f}."
            )

        # 2. Ill-conditioning Check
        if cond > 25.0 and verdict != "OSCILLATING_UNSTABLE":
            verdict = "ILL_CONDITIONED"
            rec_wd = float(self.current_wd * 2.0)
            summary += (
                f" Basin is highly anisotropic (condition number kappa = {cond:.1f}). "
                f"Recommend increasing weight decay to {rec_wd:.3f} to regularize narrow directions."
            )

        # 3. Batch Size Guidance
        if snr < 0.2:
            rec_b_scale = max(rec_b_scale, 2.0)
            summary += f" Stochastic gradient noise dominates descent (SNR = {snr:.2f} < 0.2). Increase batch size by 2x."
        elif snr > 15.0 and rec_b_scale == 1.0:
            rec_b_scale = 0.5
            summary += f" High gradient SNR ({snr:.1f}). Batch size can be halved to economize TPU compute without loss."

        return LandscapeDiagnostics(
            loss=loss,
            grad_norm=grad_norm,
            lambda_max=lambda_max,
            lambda_min=lambda_min,
            condition_number=cond,
            anisotropy_ratio=anisotropy,
            flatness_radius=flatness_radius,
            eos_margin=eos_margin,
            stochastic_snr=snr,
            stability_verdict=verdict,
            recommended_lr=rec_lr,
            recommended_weight_decay=rec_wd,
            recommended_batch_scale=rec_b_scale,
            summary=summary
        )


class SweepAdvisor:
    """Guides hyperparameter sweeps by comparing diagnostic profiles across trials."""

    def __init__(self):
        self.trial_records: List[Dict[str, Any]] = []

    def record_trial(
        self,
        trial_id: str,
        hparams: Dict[str, Any],
        diagnostics: LandscapeDiagnostics,
        eval_metric: float
    ) -> None:
        """Records a single trial's hyperparameter configuration and diagnostic profile."""
        self.trial_records.append({
            "trial_id": trial_id,
            "hparams": hparams,
            "diag": diagnostics.to_dict(),
            "eval_metric": float(eval_metric)
        })

    def recommend_next_sweep(self) -> Dict[str, Any]:
        """Synthesizes all evaluated trials and proposes a refined, high-probability parameter grid."""
        if not self.trial_records:
            return {}

        # Sort by evaluation metric (assuming lower loss / higher accuracy is better)
        sorted_trials = sorted(self.trial_records, key=lambda t: t["eval_metric"])
        best_trial = sorted_trials[0]
        best_diag = best_trial["diag"]

        suggested_lr_center = best_diag["recommended_lr"]
        suggested_wd_center = best_diag["recommended_weight_decay"]

        refined_lr_grid = [
            float(suggested_lr_center * 0.5),
            float(suggested_lr_center),
            float(suggested_lr_center * 1.5)
        ]
        refined_wd_grid = [
            float(suggested_wd_center * 0.5),
            float(suggested_wd_center),
            float(suggested_wd_center * 2.0)
        ]

        return {
            "best_trial_id": best_trial["trial_id"],
            "best_hparams": best_trial["hparams"],
            "best_eval_metric": best_trial["eval_metric"],
            "landscape_diagnosis": best_diag["summary"],
            "refined_learning_rates": refined_lr_grid,
            "refined_weight_decays": refined_wd_grid,
            "batch_scale_factor": best_diag["recommended_batch_scale"]
        }
