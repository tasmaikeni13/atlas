"""Distributed Training Script for Vision Transformer (ViT) on ImageNet-100 on Google Cloud TPUs.

Features:
- Pure Attention Vision Transformer (ViT-Small/16 or ViT-Tiny/16)
- XLA fused multi-head bidirectional attention on TPU TensorCores
- ImageNet-100 streaming pipeline with offline synthetic fallback
- Automatic ATLAS loss landscape recorder and diagnostic certification
- Multi-step AdamW with warmup and cosine decay

Usage:
    python experiments/07_train_vit_imagenet100.py --smoke_test
    python experiments/07_train_vit_imagenet100.py --batch_size 128 --learning_rate 1e-3 --epochs 90
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time
from typing import Any, Dict, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import optax

os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
jax.config.update("jax_default_matmul_precision", "highest")

from atlas.vit_imagenet import VisionTransformerImageNet, create_vit_small_imagenet, create_vit_tiny_imagenet
from atlas.imagenet100 import ImageNet100Dataset
from atlas import AtlasRecorder


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision Transformer on ImageNet-100")
    parser.add_argument("--img_size", type=int, default=224, help="Image resolution")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per TPU host")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="AdamW weight decay")
    parser.add_argument("--epochs", type=int, default=90, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps")
    parser.add_argument("--seed", type=int, default=42, help="PRNG key seed")
    parser.add_argument("--model_type", type=str, default="small", choices=["tiny", "small"], help="Model size")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick 3-step verification on synthetic data")
    parser.add_argument("--output_dir", type=str, default="runs/vit_imagenet100", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=======================================================================")
    print("  ATLAS: VISION TRANSFORMER (ViT) TRAINING ON IMAGENET-100 (TPU v4)")
    print("=======================================================================")
    print(f"Hardware: {jax.devices()} | Precision: {jax.default_backend()}")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate Model
    if args.smoke_test or args.model_type == "tiny":
        model = create_vit_tiny_imagenet(num_classes=100, img_size=args.img_size, patch_size=args.patch_size)
    else:
        model = create_vit_small_imagenet(num_classes=100, img_size=args.img_size, patch_size=args.patch_size)

    # 2. Data Pipeline
    dataset = ImageNet100Dataset(
        img_size=args.img_size,
        num_classes=100,
        split="train",
        synthetic_fallback=args.smoke_test,
        seed=args.seed
    )

    batch_size = 4 if args.smoke_test else args.batch_size
    data_stream = dataset.get_stream(batch_size=batch_size)

    # 3. Model Initialization
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    dummy_x = jnp.zeros((1, args.img_size, args.img_size, 3), dtype=jnp.float32)
    
    t0 = time.perf_counter()
    params = model.init(init_rng, dummy_x)
    
    # Parameter count
    param_leaves = jax.tree_util.tree_leaves(params)
    param_count = sum(l.size for l in param_leaves)
    print(f"Model initialized in {time.perf_counter() - t0:.2f}s with {param_count:,} parameters.")

    # 4. Optimizer with Warmup & Cosine Decay
    total_steps = 5 if args.smoke_test else (args.epochs * 1300)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=args.learning_rate,
        warmup_steps=args.warmup_steps if not args.smoke_test else 2,
        decay_steps=total_steps,
        end_value=1e-5
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay)
    )
    opt_state = optimizer.init(params)

    # 5. Loss & Step Functions
    def loss_fn(p, batch):
        x, y = batch
        logits = model.apply(p, x, deterministic=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y)
        return jnp.mean(loss)

    @jax.jit
    def train_step(p, opt_s, b):
        loss_val, grads = jax.value_and_grad(loss_fn)(p, b)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_s, loss_val, grads

    # 6. Evaluation Batches for ATLAS Landscape Probing
    eval_dataset = [dataset.generate_synthetic_batch(batch_size=batch_size) for _ in range(4)]

    # 7. ATLAS Landscape Recorder Integration
    recorder = AtlasRecorder(
        apply_fn=lambda p, b: model.apply(p, b[0], deterministic=True),
        loss_fn=lambda logits, b: jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, b[1])),
        eval_batches=eval_dataset,
        every=1 if args.smoke_test else 100,
        max_snapshots=12
    )

    # 8. Training Loop
    print(f"\nStarting training ({'Smoke Test: 3 steps' if args.smoke_test else f'{total_steps} steps'})...")
    num_steps = 3 if args.smoke_test else total_steps
    for step in range(num_steps):
        batch = next(data_stream)
        t0_step = time.perf_counter()
        params, opt_state, loss_val, grads = train_step(params, opt_state, batch)
        step_time = time.perf_counter() - t0_step
        loss_scalar = float(loss_val)

        # Record parameter update in ATLAS
        recorder.step(params, grad=grads, loss=loss_scalar)

        if step % (1 if args.smoke_test else 50) == 0:
            print(f"Step {step+1:04d}/{num_steps} | Loss: {loss_scalar:.4f} | Time: {step_time*1000:.1f}ms")

    print(f"\nTraining completed successfully. Total recorded trajectory snapshots: {len(recorder.snapshots)}")

    # 9. Certified Landscape Diagnostic Rendering
    if len(recorder.snapshots) >= 2:
        print("\nExtracting ATLAS budget-optimal landscape diagnostics on TPU...")
        report = recorder.render(
            output_dir=str(out_dir),
            budget_seconds=2.0 if args.smoke_test else 5.0,
            resolution=25 if args.smoke_test else 60,
            animate=True
        )
        print("\n" + report.summary())

    print("=======================================================================")
    print("  ViT IMAGENET-100 TRAINING & DIAGNOSTIC RUN FINISHED SUCCESSFULLY!")
    print("=======================================================================")


if __name__ == "__main__":
    main()
