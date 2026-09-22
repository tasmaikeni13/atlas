"""Train Vision Transformer (ViT) on CIFAR-10 using pure attention and record trajectory."""

from __future__ import annotations

import os
import sys
import json
import time
import pathlib
import numpy as np
import jax
import jax.numpy as jnp
import optax

from atlas.device import setup_tpu_runtime, flatten_params, unflatten_params
setup_tpu_runtime()

from atlas.models import VisionTransformer
from atlas.data import load_cifar10, BatchIterator

def main():
    print("=== Training Vision Transformer (ViT) on TPU ===")
    runs_dir = pathlib.Path("runs/vit")
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("Loading CIFAR-10 dataset...")
    train_x, train_y, test_x, test_y = load_cifar10(max_train=4000, max_test=800)
    print(f"Train size: {len(train_x)} | Test size: {len(test_x)}")

    # 2. Initialize Pure Attention ViT
    # 32x32 image with 4x4 patches -> 64 tokens, 4 layers, 4 heads, d_model=128
    vit = VisionTransformer(
        patch_size=4,
        num_classes=10,
        d_model=128,
        d_ff=256,
        num_layers=4,
        num_heads=4,
        dropout_rate=0.05
    )

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.zeros((1, 32, 32, 3), dtype=jnp.float32)
    params = vit.init(init_rng, dummy_input)
    flat_init, meta = flatten_params(params)
    print(f"ViT Initialized: {len(flat_init):,} trainable parameters (Pure Attention)")

    # 3. Setup Optimizer
    learning_rate = 3e-4
    num_epochs = 12
    batch_size = 64
    total_steps = (len(train_x) // batch_size) * num_epochs

    schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=total_steps, alpha=0.1)
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    # 4. Define JIT Loss & Step Functions
    @jax.jit
    def loss_fn(p, batch_x, batch_y, rng_key):
        logits = vit.apply(p, batch_x, deterministic=False, rngs={"dropout": rng_key})
        one_hot = jax.nn.one_hot(batch_y, 10)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch_y)
        return loss, acc

    @jax.jit
    def eval_loss_fn(p, batch_x, batch_y):
        logits = vit.apply(p, batch_x, deterministic=True)
        one_hot = jax.nn.one_hot(batch_y, 10)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch_y)
        return loss, acc

    @jax.jit
    def train_step(p, opt_s, batch_x, batch_y, rng_key):
        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, batch_x, batch_y, rng_key)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_s, loss, acc, grads

    # 5. Training Loop with Trajectory Recording
    trajectory_snapshots = [np.array(flat_init)]
    recorded_losses = []
    snapshot_steps = [0]
    global_step = 0
    record_every = 20

    t0 = time.time()
    for epoch in range(num_epochs):
        loader = BatchIterator((train_x, train_y), batch_size=batch_size, shuffle=True, seed=epoch)
        epoch_losses = []
        epoch_accs = []
        
        for bx, by in loader:
            global_step += 1
            rng, step_rng = jax.random.split(rng)
            params, opt_state, loss_val, acc_val, grads = train_step(
                params, opt_state, jnp.array(bx), jnp.array(by), step_rng
            )
            epoch_losses.append(float(loss_val))
            epoch_accs.append(float(acc_val))

            if global_step % record_every == 0:
                flat_p, _ = flatten_params(params)
                trajectory_snapshots.append(np.array(flat_p))
                recorded_losses.append(float(loss_val))
                snapshot_steps.append(global_step)

        # Eval epoch
        test_loss, test_acc = eval_loss_fn(params, jnp.array(test_x), jnp.array(test_y))
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train Loss: {np.mean(epoch_losses):.4f} | Train Acc: {np.mean(epoch_accs)*100:.1f}% | Test Acc: {float(test_acc)*100:.1f}%")

    t1 = time.time()
    print(f"Training completed in {t1 - t0:.2f}s across {global_step} steps.")

    # Save trajectory and metadata
    flat_final, _ = flatten_params(params)
    trajectory_snapshots.append(np.array(flat_final))
    snapshot_steps.append(global_step)

    traj_array = np.stack(trajectory_snapshots, axis=0)
    np.savez_compressed(
        runs_dir / "vit_trajectory.npz",
        trajectory=traj_array,
        steps=np.array(snapshot_steps),
        final_params=np.array(flat_final),
        init_params=np.array(flat_init),
    )

    metrics = {
        "model": "VisionTransformer",
        "num_params": len(flat_init),
        "train_samples": len(train_x),
        "test_samples": len(test_x),
        "num_epochs": num_epochs,
        "final_test_acc": float(test_acc),
        "final_test_loss": float(test_loss),
        "num_snapshots": len(trajectory_snapshots),
        "training_time_seconds": t1 - t0,
    }
    with open(runs_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"ViT Trajectory saved with {len(trajectory_snapshots)} checkpoints.")

if __name__ == "__main__":
    main()
