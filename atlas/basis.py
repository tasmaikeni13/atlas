"""2D Subspace construction for loss landscape analysis."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from .device import flatten_params, unflatten_params

@dataclasses.dataclass
class SubspaceBasis:
    """Orthonormal 2D affine subspace basis for loss landscape analysis."""
    u: jnp.ndarray  # (D,) orthonormal vector 1
    v: jnp.ndarray  # (D,) orthonormal vector 2
    origin: jnp.ndarray  # (D,) base model parameter coordinates (x=0, y=0)
    meta: Any  # PyTree parameter metadata
    explained_variance_ratio: float = 1.0
    singular_values: Tuple[float, float] = (1.0, 1.0)
    method: str = "pca"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "explained_variance_ratio": float(self.explained_variance_ratio),
            "singular_values": [float(s) for s in self.singular_values],
            "dimension": int(self.u.shape[0]),
        }

    def point_at(self, x: float, y: float) -> jnp.ndarray:
        """Returns the high-dimensional parameter vector at 2D coordinate (x, y)."""
        return self.origin + x * self.u + y * self.v

    def pytree_at(self, x: float, y: float) -> Any:
        """Returns the PyTree parameters at 2D coordinate (x, y)."""
        flat = self.point_at(x, y)
        return unflatten_params(flat, self.meta)

    def project_point(self, flat_param: jnp.ndarray) -> Tuple[float, float]:
        """Projects a high-dimensional parameter vector onto (x, y) plane coordinates."""
        diff = flat_param - self.origin
        x = float(jnp.dot(diff, self.u))
        y = float(jnp.dot(diff, self.v))
        return x, y


def trajectory_pca(
    trajectory: Sequence[jnp.ndarray],
    meta: Any,
    origin: Optional[jnp.ndarray] = None
) -> SubspaceBasis:
    """Computes the optimal 2D affine subspace via PCA on optimization trajectory snapshots.
    
    Uses the dual Gram matrix method (O(T^2 D)) which is exact and fast for T << D.
    """
    assert len(trajectory) >= 2, "Trajectory must contain at least 2 checkpoints for PCA"
    
    if origin is None:
        # Default origin is the final converged checkpoint
        origin = trajectory[-1]
        
    diffs = jnp.stack([theta - origin for theta in trajectory], axis=0)  # (T, D)
    T, D = diffs.shape

    # Gram matrix: K = diffs @ diffs.T (T x T)
    K = jnp.matmul(diffs, diffs.T)
    
    # Eigendecomposition of Gram matrix
    eigenvalues, eigenvectors = jnp.linalg.eigh(K)
    # Sort descending
    idx = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = jnp.maximum(eigenvalues[idx], 0.0)
    eigenvectors = eigenvectors[:, idx]

    total_var = jnp.sum(eigenvalues)
    var_2d = eigenvalues[0] + (eigenvalues[1] if T > 1 else 0.0)
    explained_ratio = float(var_2d / (total_var + 1e-12))

    # Reconstruct top 2 singular vectors in R^D: v_k = diffs.T @ e_k / sigma_k
    s0 = jnp.sqrt(jnp.maximum(eigenvalues[0], 1e-12))
    u = jnp.matmul(diffs.T, eigenvectors[:, 0]) / s0
    u = u / jnp.linalg.norm(u)

    if T > 1 and eigenvalues[1] > 1e-12:
        s1 = jnp.sqrt(eigenvalues[1])
        v = jnp.matmul(diffs.T, eigenvectors[:, 1]) / s1
        # Ensure exact orthogonality against u
        v = v - jnp.dot(v, u) * u
        v = v / jnp.linalg.norm(v)
        sing_vals = (float(s0), float(s1))
    else:
        # Generate orthogonal vector
        rng = jax.random.PRNGKey(0)
        rand_v = jax.random.normal(rng, (D,))
        rand_v = rand_v - jnp.dot(rand_v, u) * u
        v = rand_v / jnp.linalg.norm(rand_v)
        sing_vals = (float(s0), 0.0)

    return SubspaceBasis(
        u=u,
        v=v,
        origin=origin,
        meta=meta,
        explained_variance_ratio=explained_ratio,
        singular_values=sing_vals,
        method="trajectory_pca"
    )


def gradient_subspace(
    origin: jnp.ndarray,
    grad: jnp.ndarray,
    step_dir: jnp.ndarray,
    meta: Any
) -> SubspaceBasis:
    """Constructs 2D subspace from the current gradient and update step direction."""
    u = grad / (jnp.linalg.norm(grad) + 1e-12)
    v_raw = step_dir - jnp.dot(step_dir, u) * u
    norm_v = jnp.linalg.norm(v_raw)
    if norm_v < 1e-12:
        rng = jax.random.PRNGKey(1)
        v_raw = jax.random.normal(rng, u.shape)
        v_raw = v_raw - jnp.dot(v_raw, u) * u
        norm_v = jnp.linalg.norm(v_raw)
    v = v_raw / norm_v

    return SubspaceBasis(
        u=u,
        v=v,
        origin=origin,
        meta=meta,
        explained_variance_ratio=1.0,
        singular_values=(1.0, 1.0),
        method="gradient_subspace"
    )


def filter_normalized_random(
    params: Any,
    key: jax.random.PRNGKey,
    origin_flat: Optional[jnp.ndarray] = None
) -> SubspaceBasis:
    """Constructs 2D subspace using filter-normalized random directions (Li et al. 2018).
    
    Each weight matrix layer has its random perturbation normalized to match the Frobenius norm
    of that parameter layer, preventing scale distortion across layers.
    """
    flat_orig, meta = flatten_params(params) if origin_flat is None else (origin_flat, flatten_params(params)[1])
    leaves, treedef = jax.tree_util.tree_flatten(params)
    
    k1, k2 = jax.random.split(key)
    keys1 = jax.random.split(k1, len(leaves))
    keys2 = jax.random.split(k2, len(leaves))

    norm_leaves_u = []
    norm_leaves_v = []
    for leaf, ku, kv in zip(leaves, keys1, keys2):
        ru = jax.random.normal(ku, leaf.shape)
        rv = jax.random.normal(kv, leaf.shape)
        
        leaf_norm = jnp.linalg.norm(leaf)
        ru_norm = jnp.linalg.norm(ru) + 1e-12
        rv_norm = jnp.linalg.norm(rv) + 1e-12
        
        scale = jnp.where(leaf_norm > 1e-8, leaf_norm, 1.0)
        norm_leaves_u.append(ru * (scale / ru_norm))
        norm_leaves_v.append(rv * (scale / rv_norm))

    flat_u, _ = flatten_params(jax.tree_util.tree_unflatten(treedef, norm_leaves_u))
    flat_v, _ = flatten_params(jax.tree_util.tree_unflatten(treedef, norm_leaves_v))

    # Orthonormalize (u, v)
    u = flat_u / (jnp.linalg.norm(flat_u) + 1e-12)
    v_proj = flat_v - jnp.dot(flat_v, u) * u
    v = v_proj / (jnp.linalg.norm(v_proj) + 1e-12)

    return SubspaceBasis(
        u=u,
        v=v,
        origin=flat_orig,
        meta=meta,
        explained_variance_ratio=1.0,
        singular_values=(1.0, 1.0),
        method="filter_normalized_random"
    )


def project_trajectory(trajectory: Sequence[jnp.ndarray], basis: SubspaceBasis) -> np.ndarray:
    """Projects parameter trajectory into an (N, 2) array of 2D coordinates."""
    coords = [basis.project_point(theta) for theta in trajectory]
    return np.array(coords, dtype=np.float32)


def compute_subspace_capture_ratio(grads: Sequence[jnp.ndarray], basis: SubspaceBasis) -> float:
    """Computes the fraction of gradient power captured by the 2D subspace:
    
    gamma = sum_t ( (u^T g_t)^2 + (v^T g_t)^2 ) / sum_t ||g_t||^2
    """
    subspace_energy = 0.0
    total_energy = 0.0
    for g in grads:
        gu = float(jnp.dot(g, basis.u))
        gv = float(jnp.dot(g, basis.v))
        subspace_energy += (gu * gu + gv * gv)
        total_energy += float(jnp.dot(g, g))
    return float(subspace_energy / max(total_energy, 1e-12))
