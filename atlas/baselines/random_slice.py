"""Filter-Normalized Random 2D Slice Baseline (Li et al., 2018) for TPUs."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from ..device import flatten_params, unflatten_params


def filter_normalize_direction(weights: jnp.ndarray, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
    """Applies filter-wise Frobenius normalization as defined by Li et al. (2018).
    
    For each filter/row i: d_i = d_i * (||w_i||_2 / ||d_i||_2).
    """
    d = jax.random.normal(rng_key, shape=weights.shape, dtype=weights.dtype)
    if weights.ndim == 1:
        # Bias or 1D norm scale
        w_norm = jnp.linalg.norm(weights)
        d_norm = jnp.linalg.norm(d) + 1e-12
        return d * (w_norm / d_norm)
    elif weights.ndim == 2:
        # Linear layer matrix: normalize per output neuron (per row)
        w_norm = jnp.linalg.norm(weights, axis=-1, keepdims=True)
        d_norm = jnp.linalg.norm(d, axis=-1, keepdims=True) + 1e-12
        return d * (w_norm / d_norm)
    else:
        # Higher-dim tensor (e.g. conv or multi-head projection)
        axes = tuple(range(1, weights.ndim))
        w_norm = jnp.sqrt(jnp.sum(weights ** 2, axis=axes, keepdims=True))
        d_norm = jnp.sqrt(jnp.sum(d ** 2, axis=axes, keepdims=True)) + 1e-12
        return d * (w_norm / d_norm)


class FilterNormalizedRandomSlice:
    """Filter-Normalized Random 2D Plane Projection (Li et al., 2018).
    
    Builds two filter-normalized random directions d1 and d2 anchored at the final weights:
    theta(alpha, beta) = theta* + alpha * d1 + beta * d2.
    """

    def __init__(
        self,
        params: Any,
        apply_fn: Callable[[Any, Any], jnp.ndarray],
        loss_fn: Callable[[jnp.ndarray, Any], jnp.ndarray],
        seed: int = 1234
    ):
        self.params = params
        self.apply_fn = apply_fn
        self.loss_fn = loss_fn
        
        flat_theta, meta = flatten_params(params)
        self.flat_theta = flat_theta
        self.meta = meta

        rng = jax.random.PRNGKey(seed)
        rng1, rng2 = jax.random.split(rng)

        # Generate filter-normalized PyTrees
        keys1 = jax.random.split(rng1, len(jax.tree_util.tree_leaves(params)))
        keys2 = jax.random.split(rng2, len(jax.tree_util.tree_leaves(params)))
        
        d1_tree = jax.tree_util.tree_map(
            filter_normalize_direction,
            params,
            jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params), keys1)
        )
        d2_tree = jax.tree_util.tree_map(
            filter_normalize_direction,
            params,
            jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params), keys2)
        )

        def tree_norm(t):
            return jnp.sqrt(sum(jnp.sum(l**2) for l in jax.tree_util.tree_leaves(t)))

        def tree_dot(t1, t2):
            return sum(jnp.sum(l1 * l2) for l1, l2 in zip(jax.tree_util.tree_leaves(t1), jax.tree_util.tree_leaves(t2)))

        # Normalize d1
        norm_d1 = tree_norm(d1_tree) + 1e-12
        d1_tree = jax.tree_util.tree_map(lambda x: x / norm_d1, d1_tree)

        # Orthogonalize d2 against d1
        dot12 = tree_dot(d1_tree, d2_tree)
        d2_tree = jax.tree_util.tree_map(lambda l1, l2: l2 - dot12 * l1, d1_tree, d2_tree)
        norm_d2 = tree_norm(d2_tree) + 1e-12
        d2_tree = jax.tree_util.tree_map(lambda x: x / norm_d2, d2_tree)

        self.d1_tree = d1_tree
        self.d2_tree = d2_tree

        flat_d1, _ = flatten_params(d1_tree)
        flat_d2, _ = flatten_params(d2_tree)
        self.u = flat_d1
        self.v = flat_d2

        @jax.jit
        def loss_at_coord(coord: jnp.ndarray, batch: Any) -> jnp.ndarray:
            x, y = coord[0], coord[1]
            p_tree = jax.tree_util.tree_map(
                lambda p0, ul, vl: p0 + x * ul + y * vl,
                params, self.d1_tree, self.d2_tree
            )
            logits = self.apply_fn(p_tree, batch)
            return self.loss_fn(logits, batch)

        self._jit_eval = loss_at_coord

    def evaluate_coordinates(self, coords: np.ndarray, batch: Any, chunk_size: int = 16) -> np.ndarray:
        """Evaluates random plane coordinates on TPU."""
        out = []
        for pt in coords:
            c = jnp.array(pt, dtype=jnp.float32)
            loss_val = self._jit_eval(c, batch)
            out.append(float(np.array(loss_val)))
        return np.array(out, dtype=np.float32)
