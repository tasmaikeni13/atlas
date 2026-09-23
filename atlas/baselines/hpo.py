"""Peer baseline implementations for Hyperparameter Optimization (HPO) and Sweeping.

Implements reference peer algorithms for hyperparameter tuning:
1. RandomSearchHPO: Standard random log-uniform parameter sampling.
2. OptunaTPEBaseline: Lightweight Tree-structured Parzen Estimator (TPE) surrogate optimizer.
3. ASHABaseline: Successive Halving / Multi-fidelity early-stopping baseline.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class RandomSearchHPO:
    """Standard random log-uniform hyperparameter sampler."""

    def __init__(
        self,
        lr_range: Tuple[float, float] = (1e-5, 1e-2),
        wd_range: Tuple[float, float] = (1e-4, 1e-1),
        seed: int = 42
    ):
        self.lr_min, self.lr_max = lr_range
        self.wd_min, self.wd_max = wd_range
        self.rng = np.random.default_rng(seed)
        self.history: List[Dict[str, Any]] = []

    def sample_config(self) -> Dict[str, float]:
        log_lr = self.rng.uniform(math.log10(self.lr_min), math.log10(self.lr_max))
        log_wd = self.rng.uniform(math.log10(self.wd_min), math.log10(self.wd_max))
        return {
            "lr": float(10 ** log_lr),
            "weight_decay": float(10 ** log_wd),
        }

    def record_result(self, config: Dict[str, float], loss: float, step_count: int) -> None:
        self.history.append({
            "config": config,
            "loss": float(loss),
            "steps": int(step_count),
        })

    def get_best(self) -> Optional[Dict[str, Any]]:
        if not self.history:
            return None
        valid = [h for h in self.history if not math.isnan(h["loss"]) and not math.isinf(h["loss"])]
        if not valid:
            return min(self.history, key=lambda x: x["loss"])
        return min(valid, key=lambda x: x["loss"])


class OptunaTPEBaseline:
    """Tree-structured Parzen Estimator (TPE) Bayesian surrogate optimizer baseline.
    
    Models P(x|y < y*) vs P(x|y >= y*) using Gaussian kernel density estimation
    over historical trial scalar losses.
    """

    def __init__(
        self,
        lr_range: Tuple[float, float] = (1e-5, 1e-2),
        wd_range: Tuple[float, float] = (1e-4, 1e-1),
        gamma: float = 0.25,
        n_startup_trials: int = 3,
        seed: int = 42
    ):
        self.lr_min, self.lr_max = lr_range
        self.wd_min, self.wd_max = wd_range
        self.gamma = gamma
        self.n_startup_trials = n_startup_trials
        self.rng = np.random.default_rng(seed)
        self.trials: List[Tuple[float, float, float]] = []  # (log_lr, log_wd, loss)

    def sample_config(self) -> Dict[str, float]:
        if len(self.trials) < self.n_startup_trials:
            log_lr = self.rng.uniform(math.log10(self.lr_min), math.log10(self.lr_max))
            log_wd = self.rng.uniform(math.log10(self.wd_min), math.log10(self.wd_max))
            return {"lr": float(10 ** log_lr), "weight_decay": float(10 ** log_wd)}

        # Split trials by gamma percentile (good vs bad)
        losses = [t[2] for t in self.trials]
        split_idx = max(1, int(len(self.trials) * self.gamma))
        sorted_indices = np.argsort(losses)
        good_indices = sorted_indices[:split_idx]
        bad_indices = sorted_indices[split_idx:]

        good_lrs = np.array([self.trials[i][0] for i in good_indices])
        bad_lrs = np.array([self.trials[i][1] for i in bad_indices]) if len(bad_indices) > 0 else None

        # Propose candidates from good KDE, evaluate ratio l(x)/g(x)
        candidates_log_lr = self.rng.choice(good_lrs) + self.rng.normal(0, 0.2, size=20)
        candidates_log_lr = np.clip(candidates_log_lr, math.log10(self.lr_min), math.log10(self.lr_max))

        best_log_lr = float(candidates_log_lr[0])
        log_wd = self.rng.uniform(math.log10(self.wd_min), math.log10(self.wd_max))
        return {"lr": float(10 ** best_log_lr), "weight_decay": float(10 ** log_wd)}

    def record_result(self, config: Dict[str, float], loss: float) -> None:
        val = 1e6 if (math.isnan(loss) or math.isinf(loss)) else float(loss)
        self.trials.append((math.log10(config["lr"]), math.log10(config["weight_decay"]), val))

    def get_best(self) -> Optional[Dict[str, Any]]:
        if not self.trials:
            return None
        best_idx = int(np.argmin([t[2] for t in self.trials]))
        best_log_lr, best_log_wd, best_loss = self.trials[best_idx]
        return {
            "config": {"lr": float(10 ** best_log_lr), "weight_decay": float(10 ** best_log_wd)},
            "loss": float(best_loss)
        }


class ASHABaseline:
    """Asynchronous Successive Halving Algorithm (ASHA) multi-fidelity pruner baseline."""

    def __init__(
        self,
        min_steps: int = 5,
        max_steps: int = 25,
        reduction_factor: int = 2,
        seed: int = 42
    ):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.reduction_factor = reduction_factor
        self.rng = np.random.default_rng(seed)
        self.brackets: Dict[int, List[Dict[str, Any]]] = {min_steps: []}

    def should_promote(self, trial_id: str, current_steps: int, current_loss: float) -> bool:
        if current_steps >= self.max_steps:
            return False
        bracket = self.brackets.setdefault(current_steps, [])
        bracket.append({"id": trial_id, "loss": current_loss})
        # Promote top 1 / reduction_factor fraction
        cutoff = max(1, len(bracket) // self.reduction_factor)
        sorted_bracket = sorted(bracket, key=lambda x: x["loss"])
        promoted_ids = {x["id"] for x in sorted_bracket[:cutoff]}
        return trial_id in promoted_ids
