"""Smoke-scale HPO comparison with a common selection budget and final horizon.

Each method spends the same number of *planned* training steps selecting one
configuration. The selected configuration is then trained from shared initial
weights for the same number of steps and measured on one held-out batch.
Diagnostic work is included in wall time; training steps alone are not a
complete compute budget.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time
from typing import Any, Dict, Sequence

os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")

import jax
import jax.numpy as jnp
import optax

from atlas.baselines.hpo import OptunaTPEBaseline, RandomSearchHPO
from atlas.basis import trajectory_pca
from atlas.device import flatten_params
from atlas.imagenet100 import ImageNet100Dataset
from atlas.probe import JetProbe
from atlas.sweep_advisor import LandscapeDiagnosticEngine
from atlas.vit_imagenet import VisionTransformerImageNet, create_vit_tiny_imagenet


jax.config.update("jax_default_matmul_precision", "highest")


def parse_args():
    parser = argparse.ArgumentParser(description="ATLAS versus HPO smoke benchmark")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--total_budget_steps", type=int, default=60)
    parser.add_argument("--output_dir", type=str, default="runs/hpo_benchmark")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class ViTTrialRunner:
    """Reuse one compiled optimizer step across hyperparameter configurations."""

    def __init__(self, model, validation_batch):
        self.model = model
        self.validation_batch = validation_batch
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.scale_by_adam()
        )

        def loss_fn(params, batch):
            logits = model.apply(params, batch[0], deterministic=True)
            return jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, batch[1])
            )

        self.loss_fn = loss_fn
        self.eval_fn = jax.jit(loss_fn)

        @jax.jit
        def step_fn(params, state, batch, lr, wd):
            loss, grads = jax.value_and_grad(loss_fn)(params, batch)
            updates, state = self.optimizer.update(grads, state, params)
            updates = jax.tree_util.tree_map(
                lambda update, param: -lr * (update + wd * param),
                updates,
                params,
            )
            return optax.apply_updates(params, updates), state, loss

        self.step_fn = step_fn

    def run_trial(
        self,
        config: Dict[str, float],
        batches: Sequence,
        init_params,
        opt_state=None,
        collect_snapshots: bool = False,
        evaluation_batch=None,
    ) -> Dict[str, Any]:
        params = init_params
        state = self.optimizer.init(params) if opt_state is None else opt_state
        snapshots = []
        meta = None
        if collect_snapshots:
            flat, meta = flatten_params(params)
            snapshots.append(flat)

        train_losses = []
        diverged = False
        for batch in batches:
            params, state, loss = self.step_fn(
                params, state, batch, config["lr"], config["weight_decay"]
            )
            value = float(loss)
            train_losses.append(value)
            if not math.isfinite(value) or value > 50.0:
                diverged = True
                break
            if collect_snapshots:
                flat, _ = flatten_params(params)
                snapshots.append(flat)

        eval_batch = (
            self.validation_batch if evaluation_batch is None else evaluation_batch
        )
        validation_loss = 1e4 if diverged else float(self.eval_fn(params, eval_batch))
        if not math.isfinite(validation_loss):
            validation_loss = 1e4
            diverged = True
        return {
            "validation_loss": validation_loss,
            "train_losses": train_losses,
            "diverged": diverged,
            "steps_run": len(train_losses),
            "params": params,
            "opt_state": state,
            "snapshots": snapshots,
            "meta": meta,
        }


def result_record(config, selection_runs, final_run, elapsed, candidate_trials):
    """Keep selection and final costs separate for auditability."""
    return {
        "best_loss": float(final_run["validation_loss"]),
        "best_config": {key: float(value) for key, value in config.items()},
        "selection_steps": sum(run["steps_run"] for run in selection_runs),
        "final_steps": final_run["steps_run"],
        "total_training_steps": (
            sum(run["steps_run"] for run in selection_runs)
            + final_run["steps_run"]
        ),
        "selection_validation_losses": [
            float(run["validation_loss"]) for run in selection_runs
        ],
        "selection_segments": len(selection_runs),
        "diverged_trials": sum(run["diverged"] for run in selection_runs)
        + int(final_run["diverged"]),
        "total_trials": candidate_trials + 1,
        "time_seconds": float(elapsed),
    }


def main():
    args = parse_args()
    budget = 16 if args.smoke_test else args.total_budget_steps
    if budget < 16:
        raise ValueError("total_budget_steps must be at least 16")
    selection_steps = (budget // 8) * 4
    final_steps = budget - selection_steps
    trial_steps = selection_steps // 4

    img_size = 64 if args.smoke_test else 224
    batch_size = 4 if args.smoke_test else 16
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_data = ImageNet100Dataset(
        img_size=img_size, num_classes=100, split="train",
        synthetic_fallback=True, seed=args.seed,
    )
    validation_data = ImageNet100Dataset(
        img_size=img_size, num_classes=100, split="val",
        synthetic_fallback=True, seed=args.seed + 10_000,
    )
    selection_batches = [
        training_data.generate_synthetic_batch(batch_size)
        for _ in range(selection_steps)
    ]
    final_batches = [
        training_data.generate_synthetic_batch(batch_size)
        for _ in range(final_steps)
    ]
    selection_eval_batch = validation_data.generate_synthetic_batch(batch_size)
    final_eval_batch = validation_data.generate_synthetic_batch(batch_size)

    model = (
        VisionTransformerImageNet(
            num_classes=100, img_size=img_size, patch_size=16,
            d_model=64, num_layers=2, num_heads=2, mlp_dim=128,
        )
        if args.smoke_test else create_vit_tiny_imagenet(
            num_classes=100, img_size=img_size, patch_size=16
        )
    )
    init_params = model.init(
        jax.random.PRNGKey(args.seed),
        jnp.zeros((1, img_size, img_size, 3), dtype=jnp.float32),
    )
    runner = ViTTrialRunner(model, selection_eval_batch)
    results: Dict[str, Any] = {}
    initial_config = {"lr": 3e-4, "weight_decay": 0.01}
    runner.run_trial(initial_config, selection_batches[:1], init_params)

    def finish(name, config, selection_runs, started, candidate_trials):
        final_run = runner.run_trial(
            config, final_batches, init_params,
            evaluation_batch=final_eval_batch,
        )
        results[name] = result_record(
            config, selection_runs, final_run, time.perf_counter() - started,
            candidate_trials,
        )
        print(
            f"{name}: validation loss {final_run['validation_loss']:.4f}; "
            f"{results[name]['total_training_steps']} training steps; "
            f"{results[name]['time_seconds']:.2f}s"
        )

    print(f"Backend: {jax.default_backend()}; budget: {budget} steps/method")
    print(
        f"Selection: {selection_steps} steps; final: {final_steps} steps; "
        f"validation: common selection and final batches"
    )

    # ATLAS uses its entire selection budget on one exploratory trajectory.
    started = time.perf_counter()
    probe_run = runner.run_trial(
        initial_config, selection_batches, init_params, collect_snapshots=True
    )
    if probe_run["diverged"]:
        raise RuntimeError("ATLAS exploratory trajectory diverged")
    basis = trajectory_pca(
        probe_run["snapshots"], probe_run["meta"],
        origin=probe_run["snapshots"][-1],
    )

    def apply_clean(params, batch):
        return model.apply(params, batch[0], deterministic=True)

    def loss_clean(logits, batch):
        return jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, batch[1])
        )

    probe = JetProbe(apply_clean, loss_clean, basis)
    diagnostics = LandscapeDiagnosticEngine(
        probe, current_lr=initial_config["lr"],
        current_wd=initial_config["weight_decay"],
        batch_size=batch_size, min_lr=1e-5, max_lr=1e-2,
    ).analyze(selection_eval_batch)
    atlas_config = {
        "lr": diagnostics.recommended_lr,
        "weight_decay": diagnostics.recommended_weight_decay,
    }
    finish("ATLAS", atlas_config, [probe_run], started, 1)
    results["ATLAS"]["diagnostics"] = diagnostics.to_dict()

    # Random and TPE see the same training and validation examples at each
    # candidate horizon. Both get four candidate trials.
    for name, sampler_type in [
        ("Random Search", RandomSearchHPO),
        ("Optuna TPE", OptunaTPEBaseline),
    ]:
        started = time.perf_counter()
        sampler = sampler_type(seed=args.seed)
        trials = []
        for _ in range(4):
            config = sampler.sample_config()
            run = runner.run_trial(
                config, selection_batches[:trial_steps], init_params
            )
            trials.append((config, run))
            if name == "Random Search":
                sampler.record_result(
                    config, run["validation_loss"], run["steps_run"]
                )
            else:
                sampler.record_result(config, run["validation_loss"])
        best = sampler.get_best()
        finish(name, best["config"], [run for _, run in trials], started, 4)

    # Four initial candidates, two promotions, then one final promotion.
    # The optimizer state is retained at each promotion rung.
    started = time.perf_counter()
    asha_sampler = RandomSearchHPO(seed=args.seed + 1)
    first_rung = max(1, selection_steps // 8)
    candidate_states = []
    selection_runs = []
    for _ in range(4):
        config = asha_sampler.sample_config()
        run = runner.run_trial(
            config, selection_batches[:first_rung], init_params
        )
        candidate_states.append((config, run, first_rung))
        selection_runs.append(run)

    for promote_count, target_steps in [
        (2, 2 * first_rung),
        (1, selection_steps - 6 * first_rung + 2 * first_rung),
    ]:
        ranked = sorted(
            candidate_states, key=lambda item: item[1]["validation_loss"]
        )
        promoted = []
        for config, previous, completed in ranked[:promote_count]:
            run = runner.run_trial(
                config, selection_batches[completed:target_steps],
                previous["params"], previous["opt_state"],
            )
            selection_runs.append(run)
            promoted.append((config, run, target_steps))
        candidate_states = promoted
    asha_config = candidate_states[0][0]
    finish("Successive Halving", asha_config, selection_runs, started, 4)

    report = {
        "methods": results,
        "budget_steps": budget,
        "selection_budget_steps": selection_steps,
        "final_horizon_steps": final_steps,
        "seed": args.seed,
        "backend": jax.default_backend(),
        "smoke_test": args.smoke_test,
        "shared_jit_warmup_excluded_from_timing": True,
        "budget_protocol_valid": all(
            item["total_training_steps"] == budget
            for item in results.values()
        ),
        "comparison_valid": False,
        "comparison_limitations": [
            "Synthetic images and labels are independent; loss has no learnable task signal.",
            "One seed and one held-out final batch do not establish a performance advantage.",
            "ATLAS jet work is included in wall time but not converted to training steps.",
            "The successive-halving scheduler uses synchronous promotion rungs.",
        ],
    }
    report_file = out_dir / "hpo_benchmark_report.json"
    with report_file.open("w") as stream:
        json.dump(report, stream, indent=2)
    print(f"Report: {report_file}")
    if not report["budget_protocol_valid"]:
        print("A method used fewer steps because a trial diverged.")


if __name__ == "__main__":
    main()
