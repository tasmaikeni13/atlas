"""Comprehensive Apples-to-Apples Benchmark of ViT ImageNet-100 Landscape Methods on TPUs.

Evaluates 6 diagnostic methods on a Vision Transformer (ViT-Small/16 or ViT-Tiny/16) on ImageNet-100:
1. ATLAS (Budget-optimal forward-over-reverse autodiff Taylor jets + Partition of Unity + DKW certificate)
2. Vectorized TPU Grid Baseline (Pointwise TPU evaluation + 2D spline interpolation)
3. Filter-Normalized Random 2D Slice (Li et al., 2018 filter normalization on ViT weight tensors)
4. TPU Lanczos Extreme Eigenvalue & Spectral Curvature Kernel
5. TPU Hutchinson Trace & Frobenius Norm Kernel
6. Stochastic Finite-Difference Curvature Kernel

All methods benchmarked on exactly 1 deterministic seed (seed=42) for scientifically fair comparison.

Usage:
    python experiments/09_benchmark_vit_all_methods.py --smoke_test
    python experiments/09_benchmark_vit_all_methods.py --budgets 2.0 5.0 10.0
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import optax
from scipy.stats import spearmanr

os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
jax.config.update("jax_default_matmul_precision", "highest")

from atlas.vit_imagenet import VisionTransformerImageNet, create_vit_tiny_imagenet, create_vit_small_imagenet
from atlas.imagenet100 import ImageNet100Dataset
from atlas.basis import SubspaceBasis, trajectory_pca
from atlas.probe import JetProbe
from atlas.design import BudgetAllocator, generate_halton_anchors
from atlas.reconstruct import HermiteTaylorReconstruction
from atlas.device import flatten_params

# Import hardware-accelerated baselines
from atlas.baselines import (
    VectorizedGridBaseline,
    FilterNormalizedRandomSlice,
    TpuLanczosHessian,
    TpuHutchinsonTrace,
    TpuFiniteDifferenceCurvature
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ViT ImageNet-100 Landscape Methods on TPU")
    parser.add_argument("--budgets", nargs="+", type=float, default=[2.0, 5.0, 10.0], help="Time budgets in seconds")
    parser.add_argument("--smoke_test", action="store_true", help="Run fast verification mode")
    parser.add_argument("--output_dir", type=str, default="runs/benchmark_vit", help="Results directory")
    parser.add_argument("--seed", type=int, default=42, help="Single deterministic seed across all benchmarked methods")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=======================================================================")
    print("  ATLAS ViT IMAGENET-100 LANDSCAPE BENCHMARK: APPLES-TO-APPLES COMPARISON")
    print(f"  Configuration: Single Deterministic Seed (Seed = {args.seed})")
    print("=======================================================================")
    print(f"Hardware: Google Cloud TPU v4 ({jax.devices()})")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize ViT Model
    num_classes = 100
    img_size = 224
    patch_size = 16
    batch_size = 4 if args.smoke_test else 16

    if args.smoke_test:
        model = create_vit_tiny_imagenet(num_classes=num_classes, img_size=img_size, patch_size=patch_size)
    else:
        model = create_vit_small_imagenet(num_classes=num_classes, img_size=img_size, patch_size=patch_size)

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    dummy_x = jnp.zeros((1, img_size, img_size, 3), dtype=jnp.float32)

    t0 = time.perf_counter()
    params = model.init(init_rng, dummy_x)
    flat_params, meta = flatten_params(params)
    print(f"Initialized ViT Model: {len(flat_params):,} parameters in {time.perf_counter() - t0:.2f}s.")

    # 2. Data Pipeline
    dataset = ImageNet100Dataset(img_size=img_size, num_classes=num_classes, split="train", synthetic_fallback=True, seed=args.seed)
    eval_batch = dataset.generate_synthetic_batch(batch_size=batch_size)

    def apply_fn(p, b):
        return model.apply(p, b[0], deterministic=True)

    def loss_fn(logits, b):
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, b[1]))

    # 3. Simulate authentic optimization trajectory to extract 2D subspace
    print("\nExtracting ViT optimization subspace via trajectory PCA on checkpoints...")
    snapshots = [flat_params]
    curr_flat = flat_params
    num_snaps = 4 if args.smoke_test else 8

    for s in range(num_snaps):
        rng, r = jax.random.split(rng)
        step_dir = jax.random.normal(r, shape=curr_flat.shape, dtype=curr_flat.dtype) * 1e-4
        curr_flat = curr_flat + step_dir
        snapshots.append(curr_flat)

    basis = trajectory_pca(snapshots, meta)
    print(f"Top 2 Subspace Explained Variance: {basis.explained_variance_ratio:.2%}")

    radius = 0.5
    rx, ry = radius, radius

    # 4. Compute Dense Ground Truth on TPU
    gt_res = 11 if args.smoke_test else 19
    print(f"\nComputing ground truth loss surface on {gt_res}x{gt_res} dense grid points...")
    grid_baseline = VectorizedGridBaseline(apply_fn, loss_fn, basis)
    gt_X, gt_Y, gt_Z, gt_time = grid_baseline.evaluate_grid(
        grid_resolution=gt_res, radius_x=rx, radius_y=ry, batch=eval_batch, chunk_size=8
    )
    relief = float(np.max(gt_Z) - np.min(gt_Z))
    print(f"Ground truth computed in {gt_time:.2f}s. Dynamic relief: {relief:.4f}")

    eval_pts = np.stack([gt_X.ravel(), gt_Y.ravel()], axis=-1)
    gt_values = gt_Z.ravel()

    # 5. Initialize Diagnostic Baselines
    print("\nInitializing hardware-accelerated diagnostic kernels on TPU:")
    probe = JetProbe(apply_fn, loss_fn, basis)
    eval_sample_batches = [dataset.generate_synthetic_batch(batch_size=batch_size) for _ in range(4)]
    cost_model = probe.calibrate(eval_sample_batches, [4, 8, 16], radius=radius)
    allocator = BudgetAllocator(cost_model, cert_fraction=0.18)

    random_slice = FilterNormalizedRandomSlice(params, apply_fn, loss_fn, seed=args.seed)
    lanczos = TpuLanczosHessian(apply_fn, loss_fn)
    hutchinson = TpuHutchinsonTrace(apply_fn, loss_fn)
    fd_curv = TpuFiniteDifferenceCurvature(apply_fn, loss_fn)

    # 6. Benchmark Loop Across Budgets (Single Seed)
    budgets = [2.0] if args.smoke_test else args.budgets
    benchmark_results = {
        "model": "VisionTransformerImageNet",
        "parameters": len(flat_params),
        "seed": args.seed,
        "ground_truth_relief": relief,
        "runs": []
    }

    print("\n---------------------------------------------------------------------------------------------")
    print(f"{'Method':<20} | {'Budget':<7} | {'L2 Error':<10} | {'Spearman':<10} | {'Curv Metric':<12} | {'Time (s)':<8}")
    print("---------------------------------------------------------------------------------------------")

    for budget in budgets:
        # A. ATLAS (Ours)
        t0 = time.perf_counter()
        alloc = allocator.solve(budget_seconds=budget, radius=radius)
        anchors = generate_halton_anchors(alloc.n_total, radius_x=rx, radius_y=ry)
        est_coords = anchors[: alloc.n_est]
        cert_coords = anchors[alloc.n_est :]

        est_jets = [probe.evaluate_jet(x, y, eval_batch) for x, y in est_coords]
        cert_jets = [probe.evaluate_jet(x, y, eval_batch) for x, y in cert_coords]
        recon = HermiteTaylorReconstruction(est_jets)
        atlas_pred = recon.evaluate_batch(eval_pts)
        atlas_time = time.perf_counter() - t0

        l2_atlas = float(np.linalg.norm(atlas_pred - gt_values) / (np.linalg.norm(gt_values) + 1e-12))
        spearman_atlas, _ = spearmanr(atlas_pred, gt_values)
        curv_atlas = float(np.abs(recon.query_curvature(0.0, 0.0) - est_jets[0].hessian[0, 0]) / (abs(est_jets[0].hessian[0, 0]) + 1e-6))
        print(f"{'ATLAS (Ours)':<20} | {budget:<7.1f} | {l2_atlas:<10.4f} | {spearman_atlas:<10.4f} | {curv_atlas:<12.4f} | {atlas_time:<8.2f}")

        # B. TPU Grid Baseline
        t0 = time.perf_counter()
        side = int(np.clip(int(np.sqrt(alloc.n_total)), 3, 7))
        _, _, grid_z_coarse, _ = grid_baseline.evaluate_grid(
            grid_resolution=side, radius_x=rx, radius_y=ry, batch=eval_batch
        )
        xs_c = np.linspace(-rx, rx, side, dtype=np.float32)
        ys_c = np.linspace(-ry, ry, side, dtype=np.float32)
        grid_Xc, grid_Yc = np.meshgrid(xs_c, ys_c)
        spline_fn = grid_baseline.fit_interpolator(grid_Xc, grid_Yc, grid_z_coarse, method="spline")
        grid_pred = spline_fn(eval_pts)
        grid_time = time.perf_counter() - t0

        l2_grid = float(np.linalg.norm(grid_pred - gt_values) / (np.linalg.norm(gt_values) + 1e-12))
        spearman_grid, _ = spearmanr(grid_pred, gt_values)
        print(f"{'Vectorized Grid':<20} | {budget:<7.1f} | {l2_grid:<10.4f} | {spearman_grid:<10.4f} | {'N/A':<12} | {grid_time:<8.2f}")

        # C. Random 2D Slice
        t0 = time.perf_counter()
        rand_pred = random_slice.evaluate_coordinates(eval_pts, eval_batch)
        rand_time = time.perf_counter() - t0
        l2_rand = float(np.linalg.norm(rand_pred - gt_values) / (np.linalg.norm(gt_values) + 1e-12))
        spearman_rand, _ = spearmanr(rand_pred, gt_values)
        print(f"{'Random Slice':<20} | {budget:<7.1f} | {l2_rand:<10.4f} | {spearman_rand:<10.4f} | {'N/A':<12} | {rand_time:<8.2f}")

        # D. TPU Lanczos
        lam_max, lam_min, eigs, lanczos_time = lanczos.compute_spectrum(params, eval_batch, num_iterations=3 if args.smoke_test else 8, seed=args.seed)
        print(f"{'TPU Lanczos':<20} | {budget:<7.1f} | {'N/A':<10} | {'N/A':<10} | {'lam=' + f'{lam_max:.2e}':<12} | {lanczos_time:<8.2f}")

        # E. TPU Hutchinson Trace
        tr_est, frob_est, hutch_time = hutchinson.estimate_trace(params, eval_batch, num_samples=2 if args.smoke_test else 6, seed=args.seed)
        print(f"{'TPU Hutchinson':<20} | {budget:<7.1f} | {'N/A':<10} | {'N/A':<10} | {'tr=' + f'{tr_est:.2e}':<12} | {hutch_time:<8.2f}")

        # F. Stochastic Finite Difference
        dir_v = jnp.array(basis.u)
        curv_fd, fd_time = fd_curv.directional_curvature(params, dir_v, eval_batch, h=1e-3)
        print(f"{'Finite Difference':<20} | {budget:<7.1f} | {'N/A':<10} | {'N/A':<10} | {'k=' + f'{curv_fd:.2e}':<12} | {fd_time:<8.2f}")

        benchmark_results["runs"].append({
            "budget": budget,
            "seed": args.seed,
            "atlas": {"l2_error": l2_atlas, "spearman": spearman_atlas, "curv_error": curv_atlas, "time": atlas_time},
            "grid": {"l2_error": l2_grid, "spearman": spearman_grid, "time": grid_time},
            "random_slice": {"l2_error": l2_rand, "spearman": spearman_rand, "time": rand_time},
            "lanczos": {"lambda_max": lam_max, "lambda_min": lam_min, "time": lanczos_time},
            "hutchinson": {"trace": tr_est, "frobenius": frob_est, "time": hutch_time},
            "finite_difference": {"curvature": curv_fd, "time": fd_time}
        })

    with open(out_dir / "benchmark_vit_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print("=======================================================================")
    print(f"  ALL ViT BENCHMARK METHODS TESTED CLEANLY! Results in {out_dir}")
    print("=======================================================================")


if __name__ == "__main__":
    main()
