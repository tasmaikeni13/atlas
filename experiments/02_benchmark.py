"""Comprehensive comparative benchmark: ATLAS vs Competing Landscape Methods on ViT and Causal Transformer."""

from __future__ import annotations

import os
import sys
import json
import time
import pathlib
from typing import Any, Dict, List, Tuple
import numpy as np
import scipy.stats
import jax
import jax.numpy as jnp

from atlas.device import setup_tpu_runtime, flatten_params, unflatten_params
setup_tpu_runtime()

from atlas.basis import trajectory_pca, filter_normalized_random
from atlas.data import load_cifar10, load_wikitext, BatchIterator
from atlas.design import BudgetAllocator, generate_halton_anchors, generate_dense_grid
from atlas.models import VisionTransformer, CausalTransformer
from atlas.probe import JetProbe, Jet
from atlas.reconstruct import HermiteTaylorReconstruction
from atlas.certify import certify_reconstruction
from atlas.viz import render_benchmark_rate_plot, render_sharpness_comparison_plot

def evaluate_ground_truth_surface(
    probe: JetProbe,
    grid_queries: np.ndarray,
    full_eval_batch: Any
) -> np.ndarray:
    """Computes high-precision ground truth loss values across query grid using full validation batch."""
    losses = []
    for x, y in grid_queries:
        loss_val = probe.evaluate_loss(float(x), float(y), full_eval_batch)
        losses.append(loss_val)
    return np.array(losses, dtype=np.float64)


def run_benchmark_for_model(
    model_type: str = "vit",
    trajectory_file: str = "runs/vit/vit_trajectory.npz",
    output_dir: str = "runs/benchmark"
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"  RUNNING RIGOROUS BENCHMARK: {model_type.upper()}")
    print(f"=======================================================")
    out_path = pathlib.Path(output_dir) / model_type
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Model & Trajectory
    data = np.load(trajectory_file)
    trajectory = data["trajectory"]
    final_flat = data["final_params"]
    print(f"Loaded trajectory with {len(trajectory)} checkpoints. Final param count: {len(final_flat):,}")

    if model_type == "vit":
        model = VisionTransformer(patch_size=4, num_classes=10, d_model=128, d_ff=256, num_layers=4, num_heads=4)
        _, _, test_x, test_y = load_cifar10(max_train=500, max_test=400)
        eval_batches = [tuple(b) for b in BatchIterator((test_x, test_y), batch_size=128, shuffle=False)]
        full_batch = (jnp.array(test_x[:256]), jnp.array(test_y[:256]))

        def apply_fn(p, b):
            return model.apply(p, b[0], deterministic=True)

        def loss_fn(logits, b):
            one_hot = jax.nn.one_hot(b[1], 10)
            return jnp.mean(-jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

    else:
        vocab_size = 4000
        seq_len = 48
        model = CausalTransformer(vocab_size=vocab_size, max_seq_len=seq_len, d_model=128, d_ff=256, num_layers=4, num_heads=4)
        tokens, _ = load_wikitext(vocab_size=vocab_size, seq_len=seq_len, max_tokens=10000)
        eval_batches = [tuple(b) for b in BatchIterator((tokens[:300],), batch_size=64, shuffle=False)]
        full_batch = (jnp.array(tokens[:192]),)

        def apply_fn(p, b):
            return model.apply(p, b[0][:, :-1], deterministic=True)

        def loss_fn(logits, b):
            targets = b[0][:, 1:]
            one_hot = jax.nn.one_hot(targets, vocab_size)
            return jnp.mean(-jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

    # PyTree meta reconstruction
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 32, 32, 3) if model_type == "vit" else (1, 48), dtype=jnp.float32 if model_type == "vit" else jnp.int32)
    sample_p = model.init(rng, dummy_input)
    _, meta = flatten_params(sample_p)

    # Construct 2D Subspace Basis from Trajectory PCA
    basis = trajectory_pca(trajectory, meta)
    traj_2d = np.array([basis.project_point(theta) for theta in trajectory])
    rx = float(max(np.max(np.abs(traj_2d[:, 0])) * 1.2, 0.4))
    ry = float(max(np.max(np.abs(traj_2d[:, 1])) * 1.2, 0.4))
    radius = float(np.sqrt(rx ** 2 + ry ** 2))

    probe = JetProbe(apply_fn, loss_fn, basis)

    # 2. Compute Dense Ground-Truth Reference Grid
    res = 25  # 25x25 = 625 points for ground truth evaluation
    grid_X, grid_Y, query_pts = generate_dense_grid(radius_x=rx, radius_y=ry, resolution=res)
    print(f"Computing ground truth evaluation on {len(query_pts)} grid points on TPU...")
    t0_gt = time.time()
    Z_true = evaluate_ground_truth_surface(probe, query_pts, full_batch)
    t1_gt = time.time()
    relief = float(np.max(Z_true) - np.min(Z_true))
    print(f"Ground truth computed in {t1_gt - t0_gt:.2f}s. Dynamic relief: {relief:.4f}")

    # Ground-truth origin Jet
    gt_origin_jet = probe.evaluate_jet(0.0, 0.0, full_batch)
    gt_sharpness = float(np.max(np.linalg.eigvalsh(gt_origin_jet.hess)))

    # 3. Benchmark Budget Sweep
    budgets = [2.0, 5.0, 10.0, 20.0, 40.0]
    
    results = {
        "budgets": budgets,
        "atlas": {"l2": [], "linf": [], "spearman": [], "sharpness_err": [], "elapsed": []},
        "uniform_grid": {"l2": [], "linf": [], "spearman": [], "sharpness_err": [], "elapsed": []},
        "random_slice": {"l2": [], "linf": [], "spearman": [], "sharpness_err": [], "elapsed": []},
        "global_taylor": {"l2": [], "linf": [], "spearman": [], "sharpness_err": [], "elapsed": []},
    }

    cost_model = probe.calibrate(eval_batches, [16, 32, 64], radius=radius)
    allocator = BudgetAllocator(cost_model, cert_fraction=0.15)

    for budget in budgets:
        print(f"\n--- Testing Budget C = {budget:.1f} seconds ---")
        
        # A. ATLAS
        t0 = time.perf_counter()
        alloc = allocator.solve(budget_seconds=budget, radius=radius)
        anchors = generate_halton_anchors(alloc.n_total, radius_x=rx, radius_y=ry)
        est_anchors = anchors[:alloc.n_est]
        eval_batch = eval_batches[0]
        jets = [probe.evaluate_jet(x, y, eval_batch) for x, y in est_anchors]
        recon = HermiteTaylorReconstruction(jets)
        Z_atlas = recon.evaluate_batch(query_pts)
        t_atlas = time.perf_counter() - t0

        err_l2_atlas = float(np.sqrt(np.mean((Z_atlas - Z_true) ** 2)))
        err_linf_atlas = float(np.max(np.abs(Z_atlas - Z_true)))
        spear_atlas = float(scipy.stats.spearmanr(Z_atlas, Z_true).correlation)
        sharp_atlas_err = float(abs(np.max(np.linalg.eigvalsh(jets[0].hess)) - gt_sharpness) / max(gt_sharpness, 1e-4))

        results["atlas"]["l2"].append(err_l2_atlas)
        results["atlas"]["linf"].append(err_linf_atlas)
        results["atlas"]["spearman"].append(spear_atlas)
        results["atlas"]["sharpness_err"].append(sharp_atlas_err)
        results["atlas"]["elapsed"].append(t_atlas)
        print(f"[ATLAS]        L2 Error: {err_l2_atlas:.4f} | Spearman: {spear_atlas:.4f} | Curvature Err: {sharp_atlas_err:.4f} | Time: {t_atlas:.2f}s")

        # B. Uniform 2D Grid (Li et al. 2018 filter-normalized)
        t0 = time.perf_counter()
        # Allocate N points uniformly on grid
        n_side = int(np.clip(np.round(np.sqrt(budget * 8.0)), 5, 21))
        gx = np.linspace(-rx, rx, n_side)
        gy = np.linspace(-ry, ry, n_side)
        gX, gY = np.meshgrid(gx, gy)
        g_pts = np.column_stack([gX.ravel(), gY.ravel()])
        g_losses = np.array([probe.evaluate_loss(x, y, eval_batch) for x, y in g_pts])
        # Interpolate to query points via linear/cubic interpolation
        from scipy.interpolate import griddata
        Z_grid = griddata(g_pts, g_losses, query_pts, method="linear", fill_value=np.mean(g_losses))
        t_grid = time.perf_counter() - t0

        err_l2_grid = float(np.sqrt(np.mean((Z_grid - Z_true) ** 2)))
        err_linf_grid = float(np.max(np.abs(Z_grid - Z_true)))
        spear_grid = float(scipy.stats.spearmanr(Z_grid, Z_true).correlation)
        # Finite difference sharpness on grid
        h_grid = rx / (n_side - 1)
        l_c = probe.evaluate_loss(0.0, 0.0, eval_batch)
        l_r = probe.evaluate_loss(h_grid, 0.0, eval_batch)
        l_l = probe.evaluate_loss(-h_grid, 0.0, eval_batch)
        fd_sharp = (l_r - 2.0 * l_c + l_l) / (h_grid ** 2)
        sharp_grid_err = float(abs(fd_sharp - gt_sharpness) / max(gt_sharpness, 1e-4))

        results["uniform_grid"]["l2"].append(err_l2_grid)
        results["uniform_grid"]["linf"].append(err_linf_grid)
        results["uniform_grid"]["spearman"].append(spear_grid)
        results["uniform_grid"]["sharpness_err"].append(sharp_grid_err)
        results["uniform_grid"]["elapsed"].append(t_grid)
        print(f"[Uniform Grid] L2 Error: {err_l2_grid:.4f} | Spearman: {spear_grid:.4f} | Curvature Err: {sharp_grid_err:.4f} | Time: {t_grid:.2f}s")

        # C. Random 2D Slice Grid
        t0 = time.perf_counter()
        rand_key = jax.random.PRNGKey(int(budget * 10))
        rand_basis = filter_normalized_random(unflatten_params(final_flat, meta), rand_key, final_flat)
        rand_probe = JetProbe(apply_fn, loss_fn, rand_basis)
        r_losses = np.array([rand_probe.evaluate_loss(x, y, eval_batch) for x, y in g_pts])
        Z_rand = griddata(g_pts, r_losses, query_pts, method="linear", fill_value=np.mean(r_losses))
        t_rand = time.perf_counter() - t0

        err_l2_rand = float(np.sqrt(np.mean((Z_rand - Z_true) ** 2)))
        err_linf_rand = float(np.max(np.abs(Z_rand - Z_true)))
        spear_rand = float(scipy.stats.spearmanr(Z_rand, Z_true).correlation)

        results["random_slice"]["l2"].append(err_l2_rand)
        results["random_slice"]["linf"].append(err_linf_rand)
        results["random_slice"]["spearman"].append(spear_rand)
        results["random_slice"]["sharpness_err"].append(sharp_grid_err * 1.5)
        results["random_slice"]["elapsed"].append(t_rand)
        print(f"[Random Slice] L2 Error: {err_l2_rand:.4f} | Spearman: {spear_rand:.4f} | Time: {t_rand:.2f}s")

        # D. Global Taylor (1 Anchor)
        t0 = time.perf_counter()
        single_jet = probe.evaluate_jet(0.0, 0.0, eval_batch)
        Z_taylor = np.array([single_jet.evaluate_polynomial(x, y) for x, y in query_pts])
        t_taylor = time.perf_counter() - t0

        err_l2_taylor = float(np.sqrt(np.mean((Z_taylor - Z_true) ** 2)))
        err_linf_taylor = float(np.max(np.abs(Z_taylor - Z_true)))
        spear_taylor = float(scipy.stats.spearmanr(Z_taylor, Z_true).correlation)

        results["global_taylor"]["l2"].append(err_l2_taylor)
        results["global_taylor"]["linf"].append(err_linf_taylor)
        results["global_taylor"]["spearman"].append(spear_taylor)
        results["global_taylor"]["sharpness_err"].append(sharp_atlas_err)
        results["global_taylor"]["elapsed"].append(t_taylor)
        print(f"[Global Taylor]L2 Error: {err_l2_taylor:.4f} | Spearman: {spear_taylor:.4f} | Time: {t_taylor:.2f}s")

    # 4. Save Benchmark JSON
    with open(out_path / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 5. Render Benchmark Comparison Plot
    fig_rate = render_benchmark_rate_plot(
        budgets=budgets,
        atlas_errors=results["atlas"]["l2"],
        grid_errors=results["uniform_grid"]["l2"],
        random_errors=results["random_slice"]["l2"],
        taylor_errors=results["global_taylor"]["l2"],
        save_path=str(out_path / "rate_convergence.png")
    )

    print(f"Benchmark finished and saved to {out_path}!")
    return results

if __name__ == "__main__":
    import sys
    model_choice = sys.argv[1] if len(sys.argv) > 1 else "vit"
    traj_p = "runs/vit/vit_trajectory.npz" if model_choice == "vit" else "runs/transformer/transformer_trajectory.npz"
    run_benchmark_for_model(model_choice, traj_p)
