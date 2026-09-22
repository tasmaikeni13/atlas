"""Hyperparameter Sweep Diagnostic Benchmarking Script for ViT on ImageNet-100 on TPUs.

Evaluates multiple hyperparameter configurations, extracts real-time loss landscape diagnostics
via exact ATLAS Taylor jets, and synthesizes educated sweep recommendations.

Usage:
    python experiments/08_vit_sweep_diagnostics.py --smoke_test
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from typing import Any, Dict, List
import numpy as np
import jax
import jax.numpy as jnp
import optax

os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
jax.config.update("jax_default_matmul_precision", "highest")

from atlas.vit_imagenet import create_vit_tiny_imagenet
from atlas.imagenet100 import ImageNet100Dataset
from atlas.basis import SubspaceBasis, trajectory_pca
from atlas.probe import JetProbe
from atlas.sweep_advisor import LandscapeDiagnosticEngine, SweepAdvisor
from atlas.device import flatten_params


def parse_args():
    parser = argparse.ArgumentParser(description="Landscape-Guided ViT Hyperparameter Sweep Diagnostics")
    parser.add_argument("--smoke_test", action="store_true", help="Fast verification mode")
    parser.add_argument("--output_dir", type=str, default="runs/vit_sweep_diagnostics", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=======================================================================")
    print("  LANDSCAPE-GUIDED HYPERPARAMETER SWEEP DIAGNOSTICS: ViT ON IMAGENET-100")
    print("=======================================================================")
    print(f"Hardware: {jax.devices()} | Precision: {jax.default_backend()}")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Trial Configurations to Sweep
    if args.smoke_test:
        sweep_configs = [
            {"trial_id": "trial_lr_low", "lr": 1e-4, "weight_decay": 0.01},
            {"trial_id": "trial_lr_mid", "lr": 5e-4, "weight_decay": 0.05},
            {"trial_id": "trial_lr_high", "lr": 3e-3, "weight_decay": 0.01},
        ]
        steps_per_trial = 3
    else:
        sweep_configs = [
            {"trial_id": "trial_01_lr_1e-4_wd_0.01", "lr": 1e-4, "weight_decay": 0.01},
            {"trial_id": "trial_02_lr_3e-4_wd_0.05", "lr": 3e-4, "weight_decay": 0.05},
            {"trial_id": "trial_03_lr_1e-3_wd_0.05", "lr": 1e-3, "weight_decay": 0.05},
            {"trial_id": "trial_04_lr_3e-3_wd_0.01", "lr": 3e-3, "weight_decay": 0.01},
        ]
        steps_per_trial = 15

    # 2. Data Pipeline
    dataset = ImageNet100Dataset(img_size=224, num_classes=100, split="train", synthetic_fallback=True, seed=args.seed)
    batch_size = 4 if args.smoke_test else 16
    eval_batch = dataset.generate_synthetic_batch(batch_size=batch_size)

    # 3. Model Architecture
    model = create_vit_tiny_imagenet(num_classes=100, img_size=224, patch_size=16)

    advisor = SweepAdvisor()
    trial_reports = []

    print("\n---------------------------------------------------------------------------------------------")
    print(f"{'Trial ID':<22} | {'LR':<7} | {'WD':<5} | {'Loss':<6} | {'lambda_max':<10} | {'EoS Margin':<10} | {'Verdict':<20}")
    print("---------------------------------------------------------------------------------------------")

    for cfg in sweep_configs:
        trial_id = cfg["trial_id"]
        lr = cfg["lr"]
        wd = cfg["weight_decay"]

        # Initialize fresh parameters
        rng = jax.random.PRNGKey(args.seed + hash(trial_id) % 1000)
        rng, init_rng = jax.random.split(rng)
        dummy_x = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
        params = model.init(init_rng, dummy_x)
        flat_init, meta = flatten_params(params)

        # Setup trial optimizer
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr, weight_decay=wd))
        opt_state = optimizer.init(params)

        def loss_fn(p, b):
            logits = model.apply(p, b[0], deterministic=False)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, b[1]))

        @jax.jit
        def train_step(p, opt_s, b):
            l, g = jax.value_and_grad(loss_fn)(p, b)
            u, new_s = optimizer.update(g, opt_s, p)
            return optax.apply_updates(p, u), new_s, l

        # Run short trial training
        snapshots = [flat_init]
        final_loss = 0.0
        for step in range(steps_per_trial):
            batch = dataset.generate_synthetic_batch(batch_size=batch_size)
            params, opt_state, loss_val = train_step(params, opt_state, batch)
            final_loss = float(loss_val)
            flat_curr, _ = flatten_params(params)
            snapshots.append(flat_curr)

        # Build local affine 2D subspace from trajectory
        if len(snapshots) >= 2:
            basis = trajectory_pca(snapshots, meta, origin=flat_curr)
        else:
            rng, r1, r2 = jax.random.split(rng, 3)
            u_vec = np.array(jax.random.normal(r1, shape=flat_init.shape))
            u_vec /= np.linalg.norm(u_vec)
            v_vec = np.array(jax.random.normal(r2, shape=flat_init.shape))
            v_vec -= np.dot(u_vec, v_vec) * u_vec
            v_vec /= np.linalg.norm(v_vec)
            origin = np.array(flat_curr)
            basis = SubspaceBasis(u=u_vec, v=v_vec, origin=origin, meta=meta)

        # Probe landscape diagnostics using ATLAS
        def apply_clean(p, b):
            return model.apply(p, b[0], deterministic=True)

        def loss_clean(logits, b):
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, b[1]))

        probe = JetProbe(apply_clean, loss_clean, basis)
        engine = LandscapeDiagnosticEngine(probe, current_lr=lr, current_wd=wd, batch_size=batch_size)
        diag = engine.analyze(eval_batch)

        advisor.record_trial(trial_id, cfg, diag, eval_metric=final_loss)
        trial_reports.append({"trial_id": trial_id, "config": cfg, "diagnostics": diag.to_dict()})

        print(f"{trial_id:<22} | {lr:<7.1e} | {wd:<5.2f} | {final_loss:<6.3f} | {diag.lambda_max:<10.3e} | {diag.eos_margin:<10.2f} | {diag.stability_verdict:<20}")

    # Generate synthesis and next sweep recommendations
    recommendations = advisor.recommend_next_sweep()
    print("\n=======================================================================")
    print("  EDUCATED HYPERPARAMETER SWEEP GUIDANCE (FROM ATLAS LANDSCAPE ENGINE)")
    print("=======================================================================")
    print(f"Top Performing Configuration: {recommendations.get('best_trial_id')}")
    print(f"Landscape Diagnosis: {recommendations.get('landscape_diagnosis')}")
    print("\nSuggested Refined Parameter Grid for Next Sweep:")
    print(f"  -> Recommended Learning Rates:   {[f'{r:.2e}' for r in recommendations.get('refined_learning_rates', [])]}")
    print(f"  -> Recommended Weight Decays:    {[f'{w:.4f}' for w in recommendations.get('refined_weight_decays', [])]}")
    print(f"  -> Recommended Batch Multiplier: {recommendations.get('batch_scale_factor', 1.0)}x")

    # Save to disk
    with open(out_dir / "sweep_diagnostics_report.json", "w") as f:
        json.dump({"trials": trial_reports, "recommendations": recommendations}, f, indent=2)

    print(f"\nFull sweep diagnostic report written to {out_dir / 'sweep_diagnostics_report.json'}")
    print("=======================================================================")


if __name__ == "__main__":
    main()
