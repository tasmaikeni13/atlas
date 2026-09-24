"""Peer baseline implementations for Hyperparameter Optimization (HPO) and Sweeping.

Implements reference peer algorithms for hyperparameter tuning:
1. RandomSearchHPO: Standard random log-uniform parameter sampling.
2. OptunaTPEBaseline: Optuna's Tree-structured Parzen Estimator sampler.
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
    """Sequential ask/tell interface to Optuna's TPE sampler."""

    def __init__(
        self,
        lr_range: Tuple[float, float] = (1e-5, 1e-2),
        wd_range: Tuple[float, float] = (1e-4, 1e-1),
        n_startup_trials: int = 3,
        seed: int = 42
    ):
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "OptunaTPEBaseline requires the optional 'hpo' dependency"
            ) from exc

        self.lr_min, self.lr_max = lr_range
        self.wd_min, self.wd_max = wd_range
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=n_startup_trials, seed=seed
            ),
        )
        self.pending_trial = None

    def sample_config(self) -> Dict[str, float]:
        if self.pending_trial is not None:
            raise RuntimeError("Record the pending Optuna trial before sampling")
        trial = self.study.ask()
        self.pending_trial = trial
        return {
            "lr": trial.suggest_float("lr", self.lr_min, self.lr_max, log=True),
            "weight_decay": trial.suggest_float(
                "weight_decay", self.wd_min, self.wd_max, log=True
            ),
        }

    def record_result(self, config: Dict[str, float], loss: float) -> None:
        if self.pending_trial is None:
            raise RuntimeError("Sample an Optuna trial before recording a result")
        if any(
            not math.isclose(config[name], self.pending_trial.params[name])
            for name in ("lr", "weight_decay")
        ):
            raise ValueError("Result configuration differs from pending Optuna trial")
        val = 1e6 if (math.isnan(loss) or math.isinf(loss)) else float(loss)
        self.study.tell(self.pending_trial, val)
        self.pending_trial = None

    def get_best(self) -> Optional[Dict[str, Any]]:
        if not self.study.best_trials:
            return None
        return {
            "config": self.study.best_trial.params,
            "loss": float(self.study.best_value),
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
