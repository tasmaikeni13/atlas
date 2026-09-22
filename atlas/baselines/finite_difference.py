"""Hardware-accelerated Stochastic Finite-Difference Curvature Baseline for TPUs."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from ..device import flatten_params, unflatten_params


class TpuFiniteDifferenceCurvature:
    """Computes directional curvature via central finite differences on Google Cloud TPUs."""

    def __init__(
        self,
        apply_fn: Callable[[Any, Any], jnp.ndarray],
        loss_fn: Callable[[jnp.ndarray, Any], jnp.ndarray]
    ):
        self.apply_fn = apply_fn
        self.loss_fn = loss_fn

        def total_loss(p_tree: Any, batch: Any) -> jnp.ndarray:
            logits = apply_fn(p_tree, batch)
            return loss_fn(logits, batch)

        @jax.jit
        def eval_three(p_tree: Any, d_tree: Any, batch: Any, h: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            p_plus = jax.tree_util.tree_map(lambda p, d: (p + jnp.array(h, dtype=p.dtype) * d).astype(p.dtype), p_tree, d_tree)
            p_minus = jax.tree_util.tree_map(lambda p, d: (p - jnp.array(h, dtype=p.dtype) * d).astype(p.dtype), p_tree, d_tree)
            l_center = total_loss(p_tree, batch)
            l_plus = total_loss(p_plus, batch)
            l_minus = total_loss(p_minus, batch)
            return l_center, l_plus, l_minus

        self._eval_three = eval_three

    def directional_curvature(
        self,
        params: Any,
        direction: Any,
        batch: Any,
        h: float = 1e-3
    ) -> Tuple[float, float]:
        """Evaluates kappa_h = (L(theta + h*v) - 2*L(theta) + L(theta - h*v)) / h^2.
        
        Returns:
            (curvature_estimate, elapsed_seconds)
        """
        t0 = time.perf_counter()
        
        # If direction is a flat 1D array, unflatten once outside JIT
        if isinstance(direction, (np.ndarray, jnp.ndarray)) and direction.ndim == 1:
            _, meta = flatten_params(params)
            dir_tree = unflatten_params(direction, meta)
        else:
            dir_tree = direction

        # Unit norm direction
        dir_norm = jnp.sqrt(sum(jnp.sum(l**2) for l in jax.tree_util.tree_leaves(dir_tree))) + 1e-12
        unit_dir_tree = jax.tree_util.tree_map(lambda x: (x / dir_norm).astype(x.dtype), dir_tree)

        # Evaluate 3 points on TPU
        l_center, l_plus, l_minus = self._eval_three(params, unit_dir_tree, batch, h)

        curv = (l_plus - 2.0 * l_center + l_minus) / (h ** 2)
        elapsed = time.perf_counter() - t0
        return float(np.array(curv)), elapsed
