"""Empirical audit of sharpness inflation: Finite-Difference Grids vs Analytical ATLAS Jets."""

from __future__ import annotations

import os
import sys
import json
import time
import pathlib
import numpy as np
import jax
import jax.numpy as jnp

from atlas.device import setup_tpu_runtime, flatten_params, unflatten_params
setup_tpu_runtime()

from atlas.basis import trajectory_pca
from atlas.data import load_cifar10, load_wikitext, BatchIterator
from atlas.models import VisionTransformer, CausalTransformer
from atlas.probe import JetProbe
from atlas.viz import render_sharpness_comparison_plot

def audit_sharpness(model_type: str = "vit", traj_file: str = "runs/vit/vit_trajectory.npz", out_dir: str = "runs/sharpness"):
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Running Sharpness Audit on {model_type.upper()} ---")
    data = np.load(traj_file)
    trajectory = data["trajectory"]
    final_p = data["final_params"]

    if model_type == "vit":
        model = VisionTransformer(patch_size=4, num_classes=10, d_model=128, d_ff=256, num_layers=4, num_heads=4)
        _, _, test_x, test_y = load_cifar10(max_train=500, max_test=400)
        full_batch = (jnp.array(test_x[:256]), jnp.array(test_y[:256]))
        eval_samples = [tuple(b) for b in BatchIterator((test_x, test_y), batch_size=8, shuffle=True)]
        
        def apply_fn(p, b):
            return model.apply(p, b[0], deterministic=True)

        def loss_fn(logits, b):
            one_hot = jax.nn.one_hot(b[1], 10)
            return jnp.mean(-jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

        dummy = jnp.zeros((1, 32, 32, 3), dtype=jnp.float32)
    else:
        vocab_size = 4000
        seq_len = 48
        model = CausalTransformer(vocab_size=vocab_size, max_seq_len=seq_len, d_model=128, d_ff=256, num_layers=4, num_heads=4)
        tokens, _ = load_wikitext(vocab_size=vocab_size, seq_len=seq_len, max_tokens=10000)
        full_batch = (jnp.array(tokens[:192]),)
        eval_samples = [tuple(b) for b in BatchIterator((tokens[:300],), batch_size=8, shuffle=True)]

        def apply_fn(p, b):
            return model.apply(p, b[0][:, :-1], deterministic=True)

        def loss_fn(logits, b):
            targets = b[0][:, 1:]
            one_hot = jax.nn.one_hot(targets, vocab_size)
            return jnp.mean(-jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

        dummy = jnp.zeros((1, 48), dtype=jnp.int32)

    sample_p = model.init(jax.random.PRNGKey(0), dummy)
    _, meta = flatten_params(sample_p)

    basis = trajectory_pca(trajectory, meta)
    probe = JetProbe(apply_fn, loss_fn, basis)

    # 1. Ground Truth Sharpness via Full-Batch Analytical Jet on TPU
    gt_jet = probe.evaluate_jet(0.0, 0.0, full_batch)
    true_sharpness = float(np.max(np.linalg.eigvalsh(gt_jet.hess)))
    print(f"Ground-Truth Analytical Sharpness (lambda_max): {true_sharpness:.4f}")

    # 2. Finite-Difference Sharpness across mini-batch sizes B in [8, 16, 32, 64, 128, 256]
    batch_sizes = [8, 16, 32, 64, 128, 256]
    grid_estimates = []
    inflation_factors = []
    h = 0.05

    for B in batch_sizes:
        if model_type == "vit":
            sub_batch = (jnp.array(test_x[:B]), jnp.array(test_y[:B]))
        else:
            sub_batch = (jnp.array(tokens[:B]),)

        l_c = probe.evaluate_loss(0.0, 0.0, sub_batch)
        l_r = probe.evaluate_loss(h, 0.0, sub_batch)
        l_l = probe.evaluate_loss(-h, 0.0, sub_batch)
        fd_val = float((l_r - 2.0 * l_c + l_l) / (h ** 2))
        inflation = float(fd_val / max(true_sharpness, 1e-4))
        
        grid_estimates.append(fd_val)
        inflation_factors.append(inflation)
        print(f"Batch B={B:3d} | FD Sharpness: {fd_val:8.2f} | True: {true_sharpness:6.2f} | Inflation: {inflation:6.1f}x")

    # 3. Save Results
    results = {
        "model_type": model_type,
        "true_sharpness": true_sharpness,
        "batch_sizes": batch_sizes,
        "grid_estimates": grid_estimates,
        "inflation_factors": inflation_factors,
        "discretization_h": h,
    }
    with open(out_path / f"sharpness_audit_{model_type}.json", "w") as f:
        json.dump(results, f, indent=2)

    # 4. Render Figure
    fig_path = str(out_path / f"sharpness_{model_type}.png")
    render_sharpness_comparison_plot(
        batch_sizes=batch_sizes,
        grid_sharpness=grid_estimates,
        true_sharpness=true_sharpness,
        save_path=fig_path
    )
    print(f"Sharpness audit figure saved to {fig_path}")

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "vit"
    f = "runs/vit/vit_trajectory.npz" if m == "vit" else "runs/transformer/transformer_trajectory.npz"
    audit_sharpness(m, f)
