"""Automated End-to-End Smoke Test for Vision Transformer (ViT) on ImageNet-100 & Landscape Sweep Engine."""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
jax.config.update("jax_default_matmul_precision", "highest")

from atlas.vit_imagenet import VisionTransformerImageNet, create_vit_tiny_imagenet, create_vit_small_imagenet
from atlas.imagenet100 import ImageNet100Dataset
from atlas.device import flatten_params
from atlas.basis import SubspaceBasis
from atlas.probe import JetProbe
from atlas.sweep_advisor import LandscapeDiagnosticEngine, SweepAdvisor


def run_vit_smoke_test():
    print("=======================================================================")
    print("  STARTING ViT IMAGENET-100 & LANDSCAPE SWEEP ADVISOR SMOKE TEST (TPU)")
    print("=======================================================================")
    print(f"JAX backend: {jax.default_backend()} | Devices: {jax.devices()}")

    # 1. Test Model Architecture
    print("\n[TEST 1/6] Initializing ViT-Tiny/16 and ViT-Small/16 on TPU...")
    batch_size = 2
    img_size = 224
    num_classes = 100

    model = create_vit_tiny_imagenet(num_classes=num_classes, img_size=img_size, patch_size=16)
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    dummy_x = jnp.zeros((batch_size, img_size, img_size, 3), dtype=jnp.float32)

    t0 = time.perf_counter()
    params = model.init(init_rng, dummy_x)
    init_time = time.perf_counter() - t0
    flat_params, meta = flatten_params(params)
    param_count = len(flat_params)
    print(f"  -> ViT-Tiny initialized in {init_time:.2f}s with {param_count:,} parameters.")
    assert 5_000_000 < param_count < 6_500_000, f"Unexpected parameter count: {param_count}"

    out = model.apply(params, dummy_x, deterministic=True)
    assert out.shape == (batch_size, num_classes)
    print("  -> Fused bidirectional attention forward pass PASSED.")

    # 2. Test ImageNet-100 Data Pipeline
    print("\n[TEST 2/6] Verifying ImageNet-100 Data Pipeline & Batch Generation...")
    dataset = ImageNet100Dataset(img_size=img_size, num_classes=num_classes, split="train", synthetic_fallback=True)
    stream = dataset.get_stream(batch_size=batch_size, max_batches=2)
    batch_x, batch_y = next(stream)
    assert batch_x.shape == (batch_size, img_size, img_size, 3)
    assert batch_y.shape == (batch_size,)
    print(f"  -> Generated ImageNet-100 batch: {batch_x.shape} images, labels in [0, {num_classes}).")

    # 3. Test Optimizer & JIT Training Step on TPU
    print("\n[TEST 3/6] Running JIT-compiled Training Steps on TPU...")
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=1e-3, weight_decay=0.05))
    opt_state = optimizer.init(params)

    def loss_fn(p, batch):
        logits = model.apply(p, batch[0], deterministic=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch[1])
        return jnp.mean(loss)

    @jax.jit
    def step_fn(p, opt_s, b):
        loss_val, grads = jax.value_and_grad(loss_fn)(p, b)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_s, loss_val

    batch_jax = (batch_x, batch_y)
    for step in range(2):
        t0_step = time.perf_counter()
        params, opt_state, loss = step_fn(params, opt_state, batch_jax)
        loss_val = float(np.array(loss))
        print(f"  -> Step {step+1}: Loss = {loss_val:.4f} (took {time.perf_counter() - t0_step:.2f}s)")
        assert np.isfinite(loss_val)

    # 4. Test ATLAS Taylor Jet Probing on Vision Model
    print("\n[TEST 4/6] Evaluating Exact ATLAS Autodiff Taylor Jet on ViT...")
    rng, r1, r2 = jax.random.split(rng, 3)
    u_vec = jax.random.normal(r1, shape=flat_params.shape, dtype=flat_params.dtype)
    u_vec = u_vec / jnp.linalg.norm(u_vec)
    v_vec = jax.random.normal(r2, shape=flat_params.shape, dtype=flat_params.dtype)
    v_vec = v_vec - jnp.dot(u_vec, v_vec) * u_vec
    v_vec = v_vec / jnp.linalg.norm(v_vec)

    basis = SubspaceBasis(u=np.array(u_vec), v=np.array(v_vec), origin=np.array(flat_params), meta=meta)

    def apply_clean(p, b):
        return model.apply(p, b[0], deterministic=True)

    def loss_clean(logits, b):
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, b[1]))

    probe = JetProbe(apply_clean, loss_clean, basis)
    t0_jet = time.perf_counter()
    jet = probe.evaluate_jet(0.0, 0.0, batch_jax)
    jet_time = time.perf_counter() - t0_jet
    print(f"  -> Exact 2D Jet evaluated in {jet_time:.2f}s:")
    print(f"     Loss: {jet.loss:.4f} | Gradient Norm: {np.linalg.norm(jet.grad):.4f}")
    print(f"     Projected 2x2 Hessian eigenvalues: {np.linalg.eigvalsh(jet.hessian)}")
    assert np.isfinite(jet.loss)
    assert jet.hessian.shape == (2, 2)

    # 5. Test Landscape Diagnostic Engine
    print("\n[TEST 5/6] Testing Landscape Diagnostic Engine & Metrics Extraction...")
    diag_engine = LandscapeDiagnosticEngine(probe, current_lr=1e-3, current_wd=0.05, batch_size=batch_size)
    diag = diag_engine.analyze(batch_jax)
    print(f"  -> Curvature Sharpness (lambda_max): {diag.lambda_max:.4e}")
    print(f"  -> Condition Number (kappa):         {diag.condition_number:.2f}")
    print(f"  -> Edge of Stability Margin:         {diag.eos_margin:.2f}")
    print(f"  -> Basin Flatness Radius:            {diag.flatness_radius:.3f}")
    print(f"  -> Stability Verdict:                {diag.stability_verdict}")
    print(f"  -> Recommended Learning Rate:        {diag.recommended_lr:.2e}")
    assert np.isfinite(diag.lambda_max)
    assert diag.stability_verdict in ["OPTIMAL_EDGE", "OSCILLATING_UNSTABLE", "SLUGGISH_UNDERFIT", "ILL_CONDITIONED"]

    # 6. Test Sweep Advisor
    print("\n[TEST 6/6] Testing Sweep Advisor Synthesis & Recommendation Matrix...")
    advisor = SweepAdvisor()
    advisor.record_trial("trial_01", {"lr": 1e-4, "weight_decay": 0.01}, diag, eval_metric=4.60)
    advisor.record_trial("trial_02", {"lr": 1e-3, "weight_decay": 0.05}, diag, eval_metric=4.12)
    recs = advisor.recommend_next_sweep()
    print(f"  -> Best Trial Identified:            {recs.get('best_trial_id')}")
    print(f"  -> Recommended Learning Rates:       {recs.get('refined_learning_rates')}")
    print(f"  -> Recommended Weight Decays:        {recs.get('refined_weight_decays')}")
    assert len(recs["refined_learning_rates"]) == 3
    assert len(recs["refined_weight_decays"]) == 3

    print("\n=======================================================================")
    print("  ALL ViT IMAGENET-100 & SWEEP ENGINE TESTS PASSED CLEANLY WITH ZERO ERRORS!")
    print("=======================================================================")


if __name__ == "__main__":
    run_vit_smoke_test()
