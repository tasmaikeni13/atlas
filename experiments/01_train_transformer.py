"""Train Causal Language Transformer (GPT) on WikiText using pure attention and record trajectory."""

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

from atlas.models import CausalTransformer
from atlas.data import load_wikitext, BatchIterator

def main():
    print("=== Training Causal Language Transformer on TPU ===")
    runs_dir = pathlib.Path("runs/transformer")
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    vocab_size = 4000
    seq_len = 48
    print("Loading WikiText dataset...")
    tokens, vocab = load_wikitext(vocab_size=vocab_size, seq_len=seq_len, max_tokens=60000)
    
    # Split train and validation
    split_idx = int(len(tokens) * 0.85)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    print(f"Sequences: {len(train_tokens)} train, {len(val_tokens)} val | Seq Len: {seq_len}")

    # 2. Initialize Pure Attention Causal Transformer
    model = CausalTransformer(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=128,
        d_ff=256,
        num_layers=4,
        num_heads=4,
        dropout_rate=0.05
    )

    rng = jax.random.PRNGKey(101)
    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.zeros((1, seq_len), dtype=jnp.int32)
    params = model.init(init_rng, dummy_input)
    flat_init, meta = flatten_params(params)
    print(f"Causal Transformer Initialized: {len(flat_init):,} trainable parameters (Pure Attention)")

    # 3. Setup Optimizer
    learning_rate = 5e-4
    num_epochs = 12
    batch_size = 32
    total_steps = (len(train_tokens) // batch_size) * num_epochs

    schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=total_steps, alpha=0.1)
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-3)
    opt_state = optimizer.init(params)

    # 4. Define JIT Loss & Step Functions
    @jax.jit
    def loss_fn(p, batch_tok, rng_key):
        # Autoregressive next-token prediction
        inputs = batch_tok[:, :-1]
        targets = batch_tok[:, 1:]
        logits = model.apply(p, inputs, deterministic=False, rngs={"dropout": rng_key})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets).mean()
        return loss

    @jax.jit
    def eval_loss_fn(p, batch_tok):
        inputs = batch_tok[:, :-1]
        targets = batch_tok[:, 1:]
        logits = model.apply(p, inputs, deterministic=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets).mean()
        return loss

    @jax.jit
    def train_step(p, opt_s, batch_tok, rng_key):
        loss, grads = jax.value_and_grad(loss_fn)(p, batch_tok, rng_key)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_s, loss, grads

    # 5. Training Loop with Trajectory Recording
    trajectory_snapshots = [np.array(flat_init)]
    recorded_losses = []
    snapshot_steps = [0]
    global_step = 0
    record_every = 20

    t0 = time.time()
    for epoch in range(num_epochs):
        loader = BatchIterator((train_tokens,), batch_size=batch_size, shuffle=True, seed=epoch)
        epoch_losses = []

        for (btok,) in loader:
            global_step += 1
            rng, step_rng = jax.random.split(rng)
            params, opt_state, loss_val, grads = train_step(
                params, opt_state, jnp.array(btok), step_rng
            )
            epoch_losses.append(float(loss_val))

            if global_step % record_every == 0:
                flat_p, _ = flatten_params(params)
                trajectory_snapshots.append(np.array(flat_p))
                recorded_losses.append(float(loss_val))
                snapshot_steps.append(global_step)

        # Eval epoch
        val_batch = val_tokens[:min(len(val_tokens), 64)]
        val_loss = eval_loss_fn(params, jnp.array(val_batch))
        ppl = float(jnp.exp(val_loss))
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train Loss: {np.mean(epoch_losses):.4f} | Val Loss: {float(val_loss):.4f} | Perplexity: {ppl:.2f}")

    t1 = time.time()
    print(f"Training completed in {t1 - t0:.2f}s across {global_step} steps.")

    # Save trajectory and metadata
    flat_final, _ = flatten_params(params)
    trajectory_snapshots.append(np.array(flat_final))
    snapshot_steps.append(global_step)

    traj_array = np.stack(trajectory_snapshots, axis=0)
    np.savez_compressed(
        runs_dir / "transformer_trajectory.npz",
        trajectory=traj_array,
        steps=np.array(snapshot_steps),
        final_params=np.array(flat_final),
        init_params=np.array(flat_init),
    )

    metrics = {
        "model": "CausalTransformer",
        "num_params": len(flat_init),
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "num_epochs": num_epochs,
        "final_val_loss": float(val_loss),
        "final_perplexity": ppl,
        "num_snapshots": len(trajectory_snapshots),
        "training_time_seconds": t1 - t0,
    }
    with open(runs_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Causal Transformer Trajectory saved with {len(trajectory_snapshots)} checkpoints.")

if __name__ == "__main__":
    main()
