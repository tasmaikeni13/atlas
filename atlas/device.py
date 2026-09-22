"""Hardware runtime and accelerator management for Google Cloud TPUs and GPUs."""

from __future__ import annotations

import os
import sys
from typing import Any, Tuple, Dict, List
import numpy as np

def setup_tpu_runtime(single_host: bool = True) -> None:
    """Configures environment variables for Google Cloud TPU v4 Pod execution."""
    if single_host:
        os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
        os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
    os.environ.setdefault("JAX_PLATFORMS", "tpu,cpu")

setup_tpu_runtime()

import jax
import jax.numpy as jnp
from jax import tree_util

def get_tpu_device() -> Dict[str, Any]:
    """Returns runtime telemetry for available TPU accelerators."""
    devices = jax.devices()
    backend = jax.default_backend()
    is_tpu = any("tpu" in str(d).lower() for d in devices)
    
    info = {
        "backend": backend,
        "is_tpu": is_tpu,
        "device_count": len(devices),
        "devices": [str(d) for d in devices],
        "default_device": str(jax.devices()[0]),
    }
    return info

def flatten_params(pytree: Any) -> Tuple[jnp.ndarray, Any]:
    """Flattens an arbitrary PyTree of parameters into a contiguous 1D array.
    
    Returns:
        flat_vector: 1D jnp.ndarray containing all parameter values.
        treedef: structure metadata required to reconstruct the PyTree.
    """
    leaves, treedef = tree_util.tree_flatten(pytree)
    shapes = [leaf.shape for leaf in leaves]
    sizes = [leaf.size for leaf in leaves]
    flat_leaves = [jnp.reshape(leaf, (-1,)) for leaf in leaves]
    if len(flat_leaves) == 0:
        flat_vec = jnp.zeros((0,), dtype=jnp.float32)
    else:
        flat_vec = jnp.concatenate(flat_leaves, axis=0)
    meta = (treedef, shapes, sizes)
    return flat_vec, meta

def unflatten_params(flat_vec: jnp.ndarray, meta: Any) -> Any:
    """Reconstructs a PyTree parameter dictionary from a 1D vector and metadata."""
    treedef, shapes, sizes = meta
    leaves = []
    offset = 0
    for shape, size in zip(shapes, sizes):
        chunk = flat_vec[offset : offset + size]
        leaf = jnp.reshape(chunk, shape)
        leaves.append(leaf)
        offset += size
    return tree_util.tree_unflatten(treedef, leaves)

def project_vector(flat_vec: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, origin: jnp.ndarray) -> Tuple[float, float]:
    """Projects a high-dimensional parameter vector onto 2D basis (u, v) relative to origin."""
    diff = flat_vec - origin
    x = float(jnp.dot(diff, u))
    y = float(jnp.dot(diff, v))
    return x, y
