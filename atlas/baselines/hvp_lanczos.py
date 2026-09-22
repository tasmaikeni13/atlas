"""Hardware-accelerated Lanczos and Hutchinson Hessian Kernels in JAX/XLA for Google Cloud TPUs."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from ..device import flatten_params, unflatten_params


def hessian_vector_product(
    loss_fn: Callable[[Any, Any], jnp.ndarray],
    params: Any,
    v: Any,
    batch: Any
) -> Any:
    """Computes exact Hessian-Vector Product H @ v via forward-over-reverse autodiff.
    
    This compiles to exactly one reverse-mode pass followed by one forward-mode (JVP) pass
    on TPU TensorCores with zero discretization error.
    """
    def grad_fn(p):
        return jax.grad(loss_fn, argnums=0)(p, batch)
    _, hvp = jax.jvp(grad_fn, (params,), (v,))
    return hvp


def tree_norm(tree: Any) -> jnp.ndarray:
    return jnp.sqrt(sum(jnp.sum(l**2) for l in jax.tree_util.tree_leaves(tree)))


def tree_dot(t1: Any, t2: Any) -> jnp.ndarray:
    return sum(jnp.sum(a * b) for a, b in zip(jax.tree_util.tree_leaves(t1), jax.tree_util.tree_leaves(t2)))


def tree_scale(tree: Any, scalar: float | jnp.ndarray) -> Any:
    return jax.tree_util.tree_map(lambda x: (x * jnp.array(scalar, dtype=x.dtype)).astype(x.dtype), tree)


def tree_sub(t1: Any, t2: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: (x - y).astype(x.dtype), t1, t2)


class TpuLanczosHessian:
    """JAX/XLA-compiled Lanczos iteration for extreme eigenvalue and spectral density estimation on TPU."""

    def __init__(
        self,
        apply_fn: Callable[[Any, Any], jnp.ndarray],
        loss_fn: Callable[[jnp.ndarray, Any], jnp.ndarray]
    ):
        self.apply_fn = apply_fn
        self.loss_fn = loss_fn

        def total_loss(params, batch):
            logits = apply_fn(params, batch)
            return loss_fn(logits, batch)

        self._total_loss = total_loss

        # JIT-compiled HVP on PyTrees
        @jax.jit
        def tree_hvp(p_tree: Any, v_tree: Any, batch: Any) -> Any:
            return hessian_vector_product(total_loss, p_tree, v_tree, batch)

        self._jit_hvp = tree_hvp

    def compute_spectrum(
        self,
        params: Any,
        batch: Any,
        num_iterations: int = 15,
        seed: int = 42
    ) -> Tuple[float, float, np.ndarray, float]:
        """Runs Lanczos iteration to extract lambda_max, lambda_min, and tridiagonal matrix T.
        
        Returns:
            (lambda_max, lambda_min, eigenvalues, elapsed_seconds)
        """
        t0 = time.perf_counter()
        treedef = jax.tree_util.tree_structure(params)
        leaves = jax.tree_util.tree_leaves(params)

        rng = jax.random.PRNGKey(seed)
        keys = jax.random.split(rng, len(leaves))
        v_leaves = [jax.random.normal(k, shape=l.shape, dtype=jnp.float32).astype(l.dtype) for k, l in zip(keys, leaves)]
        v_curr = jax.tree_util.tree_unflatten(treedef, v_leaves)
        norm_v = tree_norm(v_curr) + 1e-12
        v_curr = tree_scale(v_curr, 1.0 / norm_v)

        alphas = []
        betas = []
        v_prev = jax.tree_util.tree_map(jnp.zeros_like, v_curr)

        for j in range(num_iterations):
            # Evaluate H @ v_curr on TPU
            w = self._jit_hvp(params, v_curr, batch)
            
            alpha = float(tree_dot(v_curr, w))
            alphas.append(alpha)

            w = tree_sub(w, tree_scale(v_curr, alpha))
            if betas:
                w = tree_sub(w, tree_scale(v_prev, betas[-1]))
            beta = float(tree_norm(w))
            
            if beta < 1e-7 or j == num_iterations - 1:
                break
            betas.append(beta)
            v_prev = v_curr
            v_curr = tree_scale(w, 1.0 / beta)

        # Construct tridiagonal matrix
        m = len(alphas)
        T = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            T[i, i] = alphas[i]
            if i < len(betas):
                T[i, i + 1] = betas[i]
                T[i + 1, i] = betas[i]

        eigenvals = np.linalg.eigvalsh(T)
        lambda_max = float(eigenvals[-1])
        lambda_min = float(eigenvals[0])
        elapsed = time.perf_counter() - t0
        return lambda_max, lambda_min, eigenvals, elapsed


class TpuHutchinsonTrace:
    """JAX/XLA Hutchinson trace and Frobenius norm estimator for Hessian curvature on TPU."""

    def __init__(
        self,
        apply_fn: Callable[[Any, Any], jnp.ndarray],
        loss_fn: Callable[[jnp.ndarray, Any], jnp.ndarray]
    ):
        def total_loss(params, batch):
            logits = apply_fn(params, batch)
            return loss_fn(logits, batch)

        @jax.jit
        def tree_hvp(p_tree: Any, v_tree: Any, batch: Any) -> Any:
            return hessian_vector_product(total_loss, p_tree, v_tree, batch)

        self._jit_hvp = tree_hvp

    def estimate_trace(
        self,
        params: Any,
        batch: Any,
        num_samples: int = 10,
        distribution: str = "rademacher",
        seed: int = 42
    ) -> Tuple[float, float, float]:
        """Computes unbiased Hutchinson trace estimator tr(H) and ||H||_F on TPU.
        
        Returns:
            (trace_estimate, frobenius_estimate, elapsed_seconds)
        """
        t0 = time.perf_counter()
        treedef = jax.tree_util.tree_structure(params)
        leaves = jax.tree_util.tree_leaves(params)
        rng = jax.random.PRNGKey(seed)

        trace_vals = []
        frob_vals = []

        for i in range(num_samples):
            rng, step_rng = jax.random.split(rng)
            keys = jax.random.split(step_rng, len(leaves))
            if distribution == "rademacher":
                v_leaves = [
                    (jax.random.bernoulli(k, 0.5, shape=l.shape).astype(l.dtype) * jnp.array(2.0, dtype=l.dtype) - jnp.array(1.0, dtype=l.dtype)).astype(l.dtype)
                    for k, l in zip(keys, leaves)
                ]
            else:
                v_leaves = [
                    jax.random.normal(k, shape=l.shape, dtype=jnp.float32).astype(l.dtype)
                    for k, l in zip(keys, leaves)
                ]
            v = jax.tree_util.tree_unflatten(treedef, v_leaves)

            # Evaluate H @ v on TPU
            hv = self._jit_hvp(params, v, batch)
            
            trace_i = float(tree_dot(v, hv))
            frob_i = float(tree_dot(hv, hv))
            trace_vals.append(trace_i)
            frob_vals.append(frob_i)

        tr_est = float(np.mean(trace_vals))
        frob_est = float(np.sqrt(np.mean(frob_vals)))
        elapsed = time.perf_counter() - t0
        return tr_est, frob_est, elapsed
