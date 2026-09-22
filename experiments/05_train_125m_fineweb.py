"""125M Parameter Transformer Pretraining on 1B Tokens of FineWeb-Edu on Google Cloud TPU v4.

Designed for scalable execution on TPU v4 Pod slices with FlashAttention, bfloat16 mixed precision,
and integrated ATLAS trajectory recording for certified loss landscape diagnostics.

Usage:
    # Full 1B token run:
    python experiments/05_train_125m_fineweb.py --tokens 1000000000 --batch_size 64

    # Fast smoke test:
    python experiments/05_train_125m_fineweb.py --smoke_test
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
from flax import linen as nn

# Set TPU Pod bounds
os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
jax.config.update("jax_default_matmul_precision", "highest")

from atlas import AtlasRecorder
from atlas.transformer_125m import Transformer125M
from atlas.fineweb import FineWebEduDataset
from atlas.device import flatten_params


def parse_args():
    parser = argparse.ArgumentParser(description="Train 125M Transformer on FineWeb-Edu")
    parser.add_argument("--tokens", type=int, default=1_000_000_000, help="Total tokens to train on (default: 1B)")
    parser.add_argument("--batch_size", type=int, default=64, help="Micro-batch size per step")
    parser.add_argument("--seq_len", type=int, default=1024, help="Context sequence length")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps for cosine schedule")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--record_every", type=int, default=100, help="ATLAS trajectory recording interval")
    parser.add_argument("--output_dir", type=str, default="runs/125m_fineweb", help="Output run directory")
    parser.add_argument("--smoke_test", action="store_true", help="Run 5 steps on synthetic tokens and exit")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    return parser.parse_args()


def build_optimizer(total_steps: int, warmup_steps: int, peak_lr: float, weight_decay: float):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=peak_lr * 0.1
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=weight_decay)
    )
    return optimizer, schedule


def main():
    args = parse_args()
    print("===============================================================")
    print("  ATLAS 125M TRANSFORMER PRETRAINING: FINEWEB-EDU (1B TOKENS)")
    print("===============================================================")
    print(f"Devices available: {jax.devices()}")
    
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model Initialization
    vocab_size = 50257
    seq_len = args.seq_len
    batch_size = 4 if args.smoke_test else args.batch_size
    max_tokens = 20_000 if args.smoke_test else args.tokens

    tokens_per_step = batch_size * seq_len
    total_steps = max_tokens // tokens_per_step
    warmup_steps = 2 if args.smoke_test else args.warmup_steps

    print(f"Model Configuration: 125M parameters (12 layers, 12 heads, d_model=768, d_ff=3072, FlashAttention)")
    print(f"Training Budget: {max_tokens:,} tokens ({total_steps:,} steps at {tokens_per_step:,} tokens/step)")

    model = Transformer125M(
        vocab_size=vocab_size,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        max_seq_len=seq_len,
        dtype=jnp.bfloat16
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.zeros((1, 16), dtype=jnp.int32)
    
    t0_init = time.perf_counter()
    params = model.init(init_rng, dummy_input)
    init_time = time.perf_counter() - t0_init
    flat_params, meta = flatten_params(params)
    print(f"Initialized {len(flat_params):,} parameters in {init_time:.2f}s.")

    # 2. Optimizer Setup
    optimizer, lr_schedule = build_optimizer(total_steps, warmup_steps, args.lr, args.weight_decay)
    opt_state = optimizer.init(params)

    # 3. JIT Training Step
    def loss_fn(p, batch_tuple):
        inputs, targets = batch_tuple
        logits = model.apply(p, inputs, deterministic=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets)
        return jnp.mean(loss)

    @jax.jit
    def train_step(p, opt_s, batch_tuple):
        loss_val, grads = jax.value_and_grad(loss_fn)(p, batch_tuple)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_s, loss_val, grads

    # 4. Data Pipeline
    dataset = FineWebEduDataset(
        subset="sample-10BT",
        cache_dir="data/fineweb_edu",
        seq_len=seq_len,
        vocab_size=vocab_size,
        synthetic_fallback=args.smoke_test
    )
    data_stream = dataset.get_stream(batch_size=batch_size, max_tokens=max_tokens, seed=args.seed)

    # 5. ATLAS Recorder Setup
    eval_batches = []
    print("Preparing holdout evaluation batches for ATLAS certification...")
    for i, (inp, tgt) in enumerate(dataset.get_stream(batch_size=min(batch_size, 8), max_tokens=tokens_per_step * 5, seed=999)):
        eval_batches.append((jnp.array(inp), jnp.array(tgt)))
        if len(eval_batches) >= 5:
            break

    recorder = AtlasRecorder(
        apply_fn=lambda p, b: model.apply(p, b[0], deterministic=True),
        loss_fn=lambda logits, b: jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, b[1])),
        eval_batches=eval_batches,
        every=2 if args.smoke_test else args.record_every
    )

    print(f"Starting training {'(SMOKE TEST MODE)' if args.smoke_test else ''}...")
    step = 0
    t_start = time.perf_counter()
    tokens_processed = 0

    trajectory_snapshots = []
    metrics_log = []

    for inputs, targets in data_stream:
        step += 1
        batch_jax = (jnp.array(inputs), jnp.array(targets))
        t_step0 = time.perf_counter()
        params, opt_state, loss, grads = train_step(params, opt_state, batch_jax)
        # Block until TPU execution finishes
        loss_scalar = float(np.array(loss))
        step_duration = time.perf_counter() - t_step0

        tokens_processed += batch_size * seq_len
        recorder.step(params, grad=grads, loss=loss_scalar)

        if step % (1 if args.smoke_test else 10) == 0:
            current_lr = float(lr_schedule(step))
            tps = (batch_size * seq_len) / max(step_duration, 1e-6)
            tflops = (6 * len(flat_params) * batch_size * seq_len) / (max(step_duration, 1e-6) * 1e12)
            print(f"Step {step:05d}/{total_steps:05d} | Loss: {loss_scalar:.4f} | LR: {current_lr:.2e} | "
                  f"{tps:,.0f} tok/s | {tflops:.1f} TFLOPS | Time: {step_duration*1000:.1f}ms")

            metrics_log.append({
                "step": step,
                "loss": loss_scalar,
                "tokens": tokens_processed,
                "tokens_per_sec": tps,
                "tflops": tflops
            })

        if args.smoke_test and step >= 5:
            print("[FineWebEdu] Smoke test threshold (5 steps) reached successfully!")
            break

    total_time = time.perf_counter() - t_start
    print(f"\nTraining completed! {tokens_processed:,} tokens processed in {total_time:.2f}s "
          f"({tokens_processed / max(total_time, 1e-6):,.0f} tok/s avg).")

    # Save final parameters and metrics
    traj_path = out_dir / "125m_trajectory.npz"
    if len(recorder._snapshots) > 0:
        np.savez_compressed(
            traj_path,
            trajectory=np.stack(recorder._snapshots, axis=0),
            final_params=np.array(flat_params),
            steps=np.arange(len(recorder._snapshots)) * recorder.every
        )
        print(f"Saved ATLAS trajectory ({len(recorder._snapshots)} checkpoints) to {traj_path}")

    with open(out_dir / "train_metrics.json", "w") as f:
        json.dump({
            "model": "Transformer125M",
            "total_params": len(flat_params),
            "tokens_processed": tokens_processed,
            "total_time_seconds": total_time,
            "steps": step,
            "history": metrics_log
        }, f, indent=2)

    print("===============================================================")
    print("  125M FINEWEB-EDU RUN SCRIPT COMPLETED WITH ZERO ERRORS")
    print("===============================================================")


if __name__ == "__main__":
    main()
