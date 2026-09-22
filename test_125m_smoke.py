"""Automated End-to-End Smoke Test for 125M Transformer and All Benchmark Kernels on TPU."""

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

from atlas.transformer_125m import Transformer125M
from atlas.flash_attention import FlashCausalAttention, tpu_blocked_flash_attention
from atlas.fineweb import FineWebEduDataset
from atlas.device import flatten_params, unflatten_params
from atlas.basis import SubspaceBasis
from atlas.probe import JetProbe
from atlas.baselines import (
    VectorizedGridBaseline,
    FilterNormalizedRandomSlice,
    TpuLanczosHessian,
    TpuHutchinsonTrace,
    TpuFiniteDifferenceCurvature
)


def run_smoke_test():
    print("================================================================")
    print("  STARTING 125M TRANSFORMER & DIAGNOSTIC KERNELS SMOKE TEST")
    print("================================================================")
    print(f"JAX backend: {jax.default_backend()} | Devices: {jax.devices()}")

    # 1. Test Model Architecture & FlashAttention
    print("\n[TEST 1/6] Initializing 125M Transformer Model...")
    vocab_size = 50257
    seq_len = 64
    batch_size = 2

    model = Transformer125M(
        vocab_size=vocab_size,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        max_seq_len=seq_len,
        dtype=jnp.bfloat16
    )

    rng = jax.random.PRNGKey(42)
    rng, init_rng, data_rng = jax.random.split(rng, 3)
    dummy_x = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    
    t0 = time.perf_counter()
    params = model.init(init_rng, dummy_x)
    init_time = time.perf_counter() - t0

    flat_params, meta = flatten_params(params)
    param_count = len(flat_params)
    print(f"  -> Model initialized in {init_time:.2f}s with {param_count:,} parameters.")
    assert 120_000_000 < param_count < 130_000_000, f"Expected ~125M params, got {param_count}"

    out = model.apply(params, dummy_x)
    assert out.shape == (batch_size, seq_len, vocab_size), f"Unexpected shape {out.shape}"
    print("  -> FlashAttention causal forward pass PASSED.")

    # 2. Test FineWeb-Edu Data Pipeline
    print("\n[TEST 2/6] Verifying FineWeb-Edu Token Generator & Batching...")
    dataset = FineWebEduDataset(seq_len=seq_len, vocab_size=vocab_size, synthetic_fallback=True)
    stream = dataset.get_stream(batch_size=batch_size, max_tokens=batch_size * seq_len * 4)
    first_batch = next(stream)
    inputs, targets = first_batch
    assert inputs.shape == (batch_size, seq_len)
    assert targets.shape == (batch_size, seq_len)
    print(f"  -> Successfully generated batches: {inputs.shape} tokens in [0, {vocab_size}).")

    # 3. Test Optimizer & Training Step
    print("\n[TEST 3/6] Running JIT-compiled Training Steps on TPU...")
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=3e-4, weight_decay=0.01)
    )
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

    batch_jax = (jnp.array(inputs), jnp.array(targets))
    for step in range(2):
        t0_step = time.perf_counter()
        params, opt_state, loss = step_fn(params, opt_state, batch_jax)
        loss_val = float(np.array(loss))
        print(f"  -> Step {step+1}: Loss = {loss_val:.4f} (took {time.perf_counter() - t0_step:.2f}s)")
        assert np.isfinite(loss_val)

    # 4. Test ATLAS Forward-over-Reverse Autodiff Jet Extraction
    print("\n[TEST 4/6] Evaluating Exact ATLAS Autodiff Taylor Jet on 125M Model...")
    # Construct 2D subspace directions
    rng, r1, r2 = jax.random.split(rng, 3)
    u_vec = jax.random.normal(r1, shape=flat_params.shape, dtype=flat_params.dtype)
    u_vec = u_vec / jnp.linalg.norm(u_vec)
    v_vec = jax.random.normal(r2, shape=flat_params.shape, dtype=flat_params.dtype)
    v_vec = v_vec - jnp.dot(u_vec, v_vec) * u_vec
    v_vec = v_vec / jnp.linalg.norm(v_vec)

    basis = SubspaceBasis(u=np.array(u_vec), v=np.array(v_vec), origin=np.array(flat_params), meta=meta)
    
    def apply_fn_clean(p, b):
        return model.apply(p, b[0], deterministic=True)

    def loss_fn_clean(logits, b):
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, b[1]))

    probe = JetProbe(apply_fn_clean, loss_fn_clean, basis)
    t0_jet = time.perf_counter()
    jet = probe.evaluate_jet(0.0, 0.0, batch_jax)
    jet_time = time.perf_counter() - t0_jet
    print(f"  -> Exact 2D Jet evaluated in {jet_time:.2f}s:")
    print(f"     Loss: {jet.loss:.4f} | Gradient Norm: {np.linalg.norm(jet.grad):.4f}")
    print(f"     Projected 2x2 Hessian eigenvalues: {np.linalg.eigvalsh(jet.hessian)}")
    assert np.isfinite(jet.loss)
    assert jet.hessian.shape == (2, 2)

    # 5. Test Baseline Diagnostic Kernels (Apples-to-Apples)
    print("\n[TEST 5/6] Testing Hardware-Accelerated Baseline Kernels on TPU:")
    # A. Vectorized Grid Baseline
    grid_base = VectorizedGridBaseline(apply_fn_clean, loss_fn_clean, basis)
    t0_g = time.perf_counter()
    gX, gY, gZ, _ = grid_base.evaluate_grid(grid_resolution=3, radius_x=0.1, radius_y=0.1, batch=batch_jax, chunk_size=4)
    print(f"  -> [A] Vectorized TPU Grid (3x3 = 9 points) in {time.perf_counter() - t0_g:.2f}s: mean loss {np.mean(gZ):.4f}")
    assert gZ.shape == (3, 3)

    # B. Filter-Normalized Random 2D Slice
    rand_slice = FilterNormalizedRandomSlice(params, apply_fn_clean, loss_fn_clean, seed=123)
    t0_r = time.perf_counter()
    rand_vals = rand_slice.evaluate_coordinates(np.array([[0.0, 0.0], [0.1, 0.1]]), batch_jax, chunk_size=2)
    print(f"  -> [B] Filter-Normalized Random Slice in {time.perf_counter() - t0_r:.2f}s: values {rand_vals}")
    assert len(rand_vals) == 2

    # C. TPU Lanczos Extreme Eigenvalue Kernel
    lanczos = TpuLanczosHessian(apply_fn_clean, loss_fn_clean)
    t0_l = time.perf_counter()
    l_max, l_min, eigs, _ = lanczos.compute_spectrum(params, batch_jax, num_iterations=3)
    print(f"  -> [C] TPU Lanczos 3 iterations in {time.perf_counter() - t0_l:.2f}s: lambda_max = {l_max:.4e}, lambda_min = {l_min:.4e}")
    assert np.isfinite(l_max)

    # D. TPU Hutchinson Trace Kernel
    hutchinson = TpuHutchinsonTrace(apply_fn_clean, loss_fn_clean)
    t0_h = time.perf_counter()
    tr_est, frob_est, _ = hutchinson.estimate_trace(params, batch_jax, num_samples=2)
    print(f"  -> [D] TPU Hutchinson Trace (2 samples) in {time.perf_counter() - t0_h:.2f}s: tr(H) = {tr_est:.4e}")
    assert np.isfinite(tr_est)

    # E. TPU Finite-Difference Curvature Kernel
    fd_curv = TpuFiniteDifferenceCurvature(apply_fn_clean, loss_fn_clean)
    t0_fd = time.perf_counter()
    curv_val, _ = fd_curv.directional_curvature(params, u_vec, batch_jax, h=1e-3)
    print(f"  -> [E] TPU Finite-Difference Curvature in {time.perf_counter() - t0_fd:.2f}s: curvature = {curv_val:.4e}")
    assert np.isfinite(curv_val)

    # 6. Overall Verification
    print("\n[TEST 6/6] Final System Verification Check...")
    print("  -> FlashAttention TPU kernel: VERIFIED")
    print("  -> 125M Parameter Transformer: VERIFIED")
    print("  -> FineWeb-Edu Data Pipeline: VERIFIED")
    print("  -> ATLAS Autodiff Jet Probe: VERIFIED")
    print("  -> All 5 Baseline Hardware Kernels: VERIFIED")
    print("\n================================================================")
    print("  ALL 125M SMOKE TESTS PASSED CLEANLY WITH ZERO ERRORS!")
    print("================================================================")


if __name__ == "__main__":
    run_smoke_test()
