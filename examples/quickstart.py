"""ATLAS Quickstart Example: 2-Line Loss Landscape Diagnostics.

This self-contained example trains a Transformer on synthetic sequence data in JAX/Flax,
attaching ATLAS in two lines to produce certified 2D/3D loss landscapes and animations.

Run with:
    python examples/quickstart.py
"""

from __future__ import annotations

import os
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as np

# Ensure TPU or CPU runtime
os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")

from atlas import AtlasRecorder


class MiniTransformer(nn.Module):
    vocab_size: int = 128
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    max_len: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T = x.shape
        pos = jnp.arange(T)[None, :]
        tok_emb = nn.Embed(self.vocab_size, self.d_model)(x)
        pos_emb = nn.Embed(self.max_len, self.d_model)(pos)
        h = tok_emb + pos_emb
        
        # Self-Attention
        mask = nn.make_causal_mask(x, dtype=bool)
        attn = nn.SelfAttention(num_heads=self.n_heads, qkv_features=self.d_model, use_bias=False)
        h = h + attn(nn.LayerNorm()(h), mask=mask)
        
        # Feed-Forward
        ff = nn.Sequential([
            nn.Dense(self.d_ff),
            nn.gelu,
            nn.Dense(self.d_model)
        ])
        h = h + ff(nn.LayerNorm()(h))
        logits = nn.Dense(self.vocab_size)(nn.LayerNorm()(h))
        return logits


def main():
    print("=== ATLAS Quickstart: Transformer Landscape Reconstruction ===")
    rng = jax.random.PRNGKey(42)
    rng, init_rng, data_rng = jax.random.split(rng, 3)

    model = MiniTransformer()
    dummy_input = jnp.zeros((1, 32), dtype=jnp.int32)
    params = model.init(init_rng, dummy_input)
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Initialized Transformer with {param_count:,} parameters.")

    def apply_fn(p, b):
        return model.apply(p, b[0])

    def loss_fn(logits, b):
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, b[1]))

    # Synthetic training batches
    num_batches = 40
    batch_size = 32
    seq_len = 32
    batches = []
    for _ in range(num_batches):
        data_rng, r1, r2 = jax.random.split(data_rng, 3)
        inp = jax.random.randint(r1, (batch_size, seq_len), 0, 128)
        tgt = jax.random.randint(r2, (batch_size, seq_len), 0, 128)
        batches.append((inp, tgt))

    eval_batches = batches[:5]

    # --- ATLAS LINE 1: Initialize Recorder ---
    recorder = AtlasRecorder(
        apply_fn=apply_fn,
        loss_fn=loss_fn,
        eval_batches=eval_batches,
        every=5
    )

    optimizer = optax.adamw(learning_rate=3e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch):
        x, y = batch
        def forward(p):
            out = model.apply(p, x)
            return loss_fn(out, batch)
        loss, grads = jax.value_and_grad(forward)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, grads

    print("Training Transformer for 50 steps...")
    for step in range(50):
        batch = batches[step % num_batches]
        params, opt_state, loss, grads = train_step(params, opt_state, batch)
        # --- ATLAS LINE 2: Record parameter update ---
        recorder.step(params, grad=grads, loss=float(loss))
        if step % 10 == 0:
            print(f"  Step {step:02d} | Loss: {float(loss):.4f}")

    print("\n--- ATLAS: Performing Budget-Optimal Reconstruction & Certification ---")
    report = recorder.render(
        output_dir="atlas_quickstart",
        budget_seconds=3.0,
        resolution=60,
        animate=False
    )
    print("\n" + report.summary())
    print("\nGenerated diagnostic assets in 'atlas_quickstart/':")
    print(f"  - 2D Certified Contour: {report.figure_2d}")
    print(f"  - 3D Loss Basin:       {report.figure_3d}")
    print(f"  - Statistical Certificate: {report.certificate_plot}")


if __name__ == "__main__":
    main()
