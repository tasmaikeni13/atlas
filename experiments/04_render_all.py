"""Renders all final publication landscapes, 3D basins, certificates, and animated GIFs for ViT and Transformer."""

from __future__ import annotations

import os
import sys
import json
import time
import pathlib
import shutil
import numpy as np
import jax
import jax.numpy as jnp

from atlas.device import setup_tpu_runtime, flatten_params, unflatten_params
setup_tpu_runtime()

from atlas.basis import trajectory_pca
from atlas.data import load_cifar10, load_wikitext, BatchIterator
from atlas.design import BudgetAllocator, generate_halton_anchors, generate_dense_grid
from atlas.models import VisionTransformer, CausalTransformer
from atlas.probe import JetProbe
from atlas.reconstruct import HermiteTaylorReconstruction
from atlas.certify import certify_reconstruction
from atlas.viz import (
    render_landscape_2d,
    render_landscape_3d,
    render_certificate_plot,
    create_landscape_gif,
)

def render_model_pipeline(model_type: str = "vit", traj_file: str = "runs/vit/vit_trajectory.npz", figures_dir: str = "figures"):
    fig_path = pathlib.Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"  RENDERING FULL ATLAS SUITE: {model_type.upper()}")
    print(f"=======================================================")

    # 1. Load Data & Trajectory
    data = np.load(traj_file)
    trajectory = data["trajectory"]
    final_flat = data["final_params"]
    print(f"Trajectory checkpoints: {len(trajectory)} | Model parameters: {len(final_flat):,}")

    if model_type == "vit":
        model = VisionTransformer(patch_size=4, num_classes=10, d_model=128, d_ff=256, num_layers=4, num_heads=4)
        _, _, test_x, test_y = load_cifar10(max_train=500, max_test=600)
        eval_batch = (jnp.array(test_x[:256]), jnp.array(test_y[:256]))
        eval_batches = [tuple(b) for b in BatchIterator((test_x, test_y), batch_size=64, shuffle=False)]

        def apply_fn(p, b):
            return model.apply(p, b[0], deterministic=True)

        def loss_fn(logits, b):
            one_hot = jax.nn.one_hot(b[1], 10)
            return jnp.mean(-jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

        dummy = jnp.zeros((1, 32, 32, 3), dtype=jnp.float32)
        model_title = "Vision Transformer (ViT) on CIFAR-10"
    else:
        vocab_size = 4000
        seq_len = 48
        model = CausalTransformer(vocab_size=vocab_size, max_seq_len=seq_len, d_model=128, d_ff=256, num_layers=4, num_heads=4)
        tokens, _ = load_wikitext(vocab_size=vocab_size, seq_len=seq_len, max_tokens=15000)
        eval_batch = (jnp.array(tokens[:192]),)
        eval_batches = [tuple(b) for b in BatchIterator((tokens[:300],), batch_size=48, shuffle=False)]

        def apply_fn(p, b):
            return model.apply(p, b[0][:, :-1], deterministic=True)

        def loss_fn(logits, b):
            targets = b[0][:, 1:]
            one_hot = jax.nn.one_hot(targets, vocab_size)
            return jnp.mean(-jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

        dummy = jnp.zeros((1, 48), dtype=jnp.int32)
        model_title = "Causal Language Transformer on WikiText"

    sample_p = model.init(jax.random.PRNGKey(0), dummy)
    _, meta = flatten_params(sample_p)

    # 2. Build Subspace Basis from Trajectory PCA
    basis = trajectory_pca(trajectory, meta)
    traj_2d = np.array([basis.project_point(theta) for theta in trajectory])
    rx = float(max(np.max(np.abs(traj_2d[:, 0])) * 1.25, 0.45))
    ry = float(max(np.max(np.abs(traj_2d[:, 1])) * 1.25, 0.45))
    radius = float(np.sqrt(rx ** 2 + ry ** 2))
    print(f"Bounding domain: [-{rx:.3f}, {rx:.3f}] x [-{ry:.3f}, {ry:.3f}] (Radius: {radius:.3f})")

    # 3. Probe Setup & Calibration
    probe = JetProbe(apply_fn, loss_fn, basis)
    cost_model = probe.calibrate(eval_batches, [16, 32, 64], radius=radius)
    print(f"Calibrated Cost: tau={cost_model.tau*1000:.2f}ms, kappa={cost_model.kappa*1e6:.2f}us/ex, M3={cost_model.m3:.2f}")

    # 4. Budget Allocation (Allocate 25s budget)
    allocator = BudgetAllocator(cost_model, cert_fraction=0.18)
    allocation = allocator.solve(budget_seconds=25.0, radius=radius)
    print(f"Optimal Allocation: {allocation.n_est} estimation anchors + {allocation.n_cert} cert anchors (Batch {allocation.batch_size})")

    # 5. Generate Anchors & Evaluate Jets on TPU
    anchors = generate_halton_anchors(allocation.n_total, radius_x=rx, radius_y=ry)
    est_anchors = anchors[:allocation.n_est]
    cert_anchors = anchors[allocation.n_est:]

    print(f"Evaluating {len(est_anchors)} analytical Taylor jets on TPU...")
    t0_jets = time.perf_counter()
    est_jets = [probe.evaluate_jet(x, y, eval_batch) for x, y in est_anchors]
    cert_jets = [probe.evaluate_jet(x, y, eval_batch) for x, y in cert_anchors]
    t1_jets = time.perf_counter()
    print(f"Jets evaluated in {t1_jets - t0_jets:.2f}s.")

    # 6. Hermite-Taylor Partition of Unity Reconstruction
    recon = HermiteTaylorReconstruction(est_jets)
    resolution = 90
    grid_X, grid_Y, query_pts = generate_dense_grid(radius_x=rx, radius_y=ry, resolution=resolution)
    print(f"Evaluating continuous surface on {len(query_pts)} points...")
    Z = recon.evaluate_batch(query_pts).reshape(grid_X.shape)

    analysis = recon.analyze_geometry(grid_X, grid_Y)
    print(f"Geometry: Condition Number={analysis.origin_condition_number:.2f} | Relief={analysis.surface_relief:.4f}")

    # 7. Statistical Error Certification
    certificate = certify_reconstruction(
        reconstruction=recon,
        cert_jets=cert_jets,
        surface_relief=analysis.surface_relief,
        confidence_level=0.95
    )
    print(f"[Certificate] {certificate.summary()}")

    # 8. Render Figures
    suffix = "vit" if model_type == "vit" else "transformer"
    fig2d_out = str(fig_path / f"landscape_{suffix}.png")
    fig3d_out = str(fig_path / f"landscape_3d_{suffix}.png")
    cert_out = str(fig_path / f"certificate_{suffix}.png")
    gif_out = str(fig_path / f"landscape_{suffix}.gif")

    print(f"Rendering 2D landscape to {fig2d_out}...")
    render_landscape_2d(
        X=grid_X, Y=grid_Y, Z=Z,
        trajectory_2d=traj_2d,
        anchors=est_anchors,
        cert_points=cert_anchors,
        analysis=analysis,
        certificate=certificate,
        title=f"ATLAS Certified Loss Landscape: {model_title}",
        save_path=fig2d_out
    )

    print(f"Rendering 3D landscape to {fig3d_out}...")
    render_landscape_3d(
        X=grid_X, Y=grid_Y, Z=Z,
        trajectory_2d=traj_2d,
        title=f"ATLAS 3D Reconstructed Basin: {model_title}",
        save_path=fig3d_out
    )

    print(f"Rendering certificate distribution to {cert_out}...")
    residuals = np.abs(recon.evaluate_batch(cert_anchors) - np.array([j.loss for j in cert_jets]))
    render_certificate_plot(residuals=residuals, certificate=certificate, save_path=cert_out)

    print(f"Rendering animated trajectory GIF to {gif_out}...")
    create_landscape_gif(
        X=grid_X, Y=grid_Y, Z=Z,
        trajectory_2d=traj_2d,
        output_path=gif_out,
        fps=6,
        title_prefix=f"ATLAS Trajectory: {suffix.upper()}"
    )

    # Save summary report
    report_dict = {
        "model_type": model_type,
        "certificate": certificate.to_dict(),
        "allocation": allocation.to_dict(),
        "analysis": analysis.to_dict(),
        "cost_model": cost_model.to_dict(),
    }
    with open(fig_path / f"report_{suffix}.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"Completed ATLAS pipeline for {model_type.upper()}!")

if __name__ == "__main__":
    render_model_pipeline("vit", "runs/vit/vit_trajectory.npz")
    render_model_pipeline("transformer", "runs/transformer/transformer_trajectory.npz")
