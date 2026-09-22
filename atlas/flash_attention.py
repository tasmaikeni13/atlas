"""High-performance FlashAttention and hardware-optimized attention kernels for Google Cloud TPUs."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn


def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Applies Rotary Position Embedding (RoPE) to Query and Key tensors.
    
    Args:
        xq: (batch, seq_len, num_heads, head_dim)
        xk: (batch, seq_len, num_heads, head_dim)
        cos: (1, seq_len, 1, head_dim)
        sin: (1, seq_len, 1, head_dim)
    """
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return jnp.concatenate((-x2, x1), axis=-1)

    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Precomputes cosine and sine frequency tables for RoPE."""
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)  # (max_seq_len, head_dim // 2)
    freqs = jnp.repeat(freqs, 2, axis=-1)  # (max_seq_len, head_dim)
    cos = jnp.cos(freqs)[None, :, None, :]  # (1, seq_len, 1, head_dim)
    sin = jnp.sin(freqs)[None, :, None, :]
    return cos, sin


def tpu_blocked_flash_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    block_size: int = 128,
    causal: bool = True
) -> jnp.ndarray:
    """Exact Blocked FlashAttention kernel in JAX for Google Cloud TPU v4.
    
    Implements the online softmax algorithm (Dao et al. 2022) with block tiling,
    streaming over Key/Value blocks using jax.lax.scan to maintain O(block_size)
    SRAM residency and avoid materializing the full O(seq_len^2) attention matrix.
    
    Args:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_heads, seq_len, head_dim)
        v: (batch, num_heads, seq_len, head_dim)
        block_size: Tiling chunk size (default 128 for TPU v4 matrix units)
        causal: Whether to apply causal autoregressive triangular masking
    """
    B, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)

    # Pad sequence to multiple of block_size if necessary
    num_blocks = math.ceil(S / block_size)
    pad_len = num_blocks * block_size - S
    if pad_len > 0:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
    
    total_s = num_blocks * block_size

    # Reshape into blocks: (num_blocks, B, H, block_size, D)
    q_blocks = jnp.reshape(q, (B, H, num_blocks, block_size, D)).transpose(2, 0, 1, 3, 4)
    k_blocks = jnp.reshape(k, (B, H, num_blocks, block_size, D)).transpose(2, 0, 1, 3, 4)
    v_blocks = jnp.reshape(v, (B, H, num_blocks, block_size, D)).transpose(2, 0, 1, 3, 4)

    def process_query_block(r_idx, q_i):
        # q_i is (B, H, block_size, D)
        # Initialize running accumulators:
        # m_i: running max (B, H, block_size, 1)
        # l_i: running sum of exp (B, H, block_size, 1)
        # o_i: running unnormalized output (B, H, block_size, D)
        m_0 = jnp.full((B, H, block_size, 1), -1e9, dtype=jnp.float32)
        l_0 = jnp.zeros((B, H, block_size, 1), dtype=jnp.float32)
        o_0 = jnp.zeros((B, H, block_size, D), dtype=jnp.float32)

        def step_kv(carry, kv_tuple):
            c_idx, k_j, v_j = kv_tuple
            m_prev, l_prev, o_prev = carry

            # If causal and key block is entirely in future, skip computation
            # Compute raw scores: (B, H, block_size, block_size)
            scores = jnp.einsum("bhid,bhjd->bhij", q_i, k_j) * scale

            if causal:
                # Mask within block if on diagonal
                q_indices = r_idx * block_size + jnp.arange(block_size)[:, None]
                k_indices = c_idx * block_size + jnp.arange(block_size)[None, :]
                mask = q_indices >= k_indices
                scores = jnp.where(mask[None, None, :, :], scores, -1e9)

            # Online softmax update
            m_curr = jnp.maximum(m_prev, jnp.max(scores, axis=-1, keepdims=True))
            # Shifted exponentials
            exp_scores = jnp.exp(scores - m_curr)
            # Rescale previous sum
            alpha = jnp.exp(m_prev - m_curr)
            l_curr = alpha * l_prev + jnp.sum(exp_scores, axis=-1, keepdims=True)

            # Update output accumulator: o = alpha * o_prev + exp_scores @ v_j
            p_v = jnp.einsum("bhij,bhjd->bhid", exp_scores.astype(v_j.dtype), v_j)
            o_curr = alpha * o_prev + p_v

            # If block is purely in future, keep carry unchanged
            if causal:
                active = (c_idx <= r_idx)
                carry_next = (
                    jnp.where(active, m_curr, m_prev),
                    jnp.where(active, l_curr, l_prev),
                    jnp.where(active, o_curr, o_prev)
                )
            else:
                carry_next = (m_curr, l_curr, o_curr)

            return carry_next, None

        c_indices = jnp.arange(num_blocks)
        (m_final, l_final, o_final), _ = jax.lax.scan(
            step_kv, (m_0, l_0, o_0), (c_indices, k_blocks, v_blocks)
        )
        # Normalize output: o_i = o_i / l_i
        out_block = o_final / jnp.maximum(l_final, 1e-6)
        return out_block

    # Scan or loop over query blocks
    r_indices = jnp.arange(num_blocks)
    out_blocks = []
    for r in range(num_blocks):
        out_b = process_query_block(r, q_blocks[r])
        out_blocks.append(out_b)

    # Concatenate back to (B, H, total_s, D)
    out = jnp.concatenate(out_blocks, axis=2)
    # Remove padding if added
    if pad_len > 0:
        out = out[:, :, :S, :]
    return out.astype(q.dtype)


def fused_causal_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    use_flash: bool = True
) -> jnp.ndarray:
    """High-level causal attention router with TPU FlashAttention and XLA remat support."""
    if use_flash:
        # Check if sequence length warrants tiled FlashAttention
        if q.shape[2] >= 256:
            return tpu_blocked_flash_attention(q, k, v, block_size=128, causal=True)

    # Standard fused dot-product attention with remat
    B, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)
    else:
        causal_mask = jnp.tril(jnp.ones((S, S), dtype=bool))[None, None, :, :]
        scores = jnp.where(causal_mask, scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhij,bhjd->bhid", weights, v)
    return out


class FlashCausalAttention(nn.Module):
    """Flax Linen Module for Flash Multi-Head Causal Self-Attention."""
    d_model: int
    num_heads: int
    dtype: jnp.dtype = jnp.float32
    use_rope: bool = True
    max_seq_len: int = 2048

    def setup(self):
        assert self.d_model % self.num_heads == 0
        self.head_dim = self.d_model // self.num_heads
        self.q_proj = nn.Dense(self.d_model, use_bias=False, dtype=self.dtype)
        self.k_proj = nn.Dense(self.d_model, use_bias=False, dtype=self.dtype)
        self.v_proj = nn.Dense(self.d_model, use_bias=False, dtype=self.dtype)
        self.out_proj = nn.Dense(self.d_model, use_bias=False, dtype=self.dtype)
        
        if self.use_rope:
            cos, sin = precompute_rope_freqs(self.head_dim, self.max_seq_len)
            self.cos = cos
            self.sin = sin

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        B, S, D = x.shape
        q = self.q_proj(x).reshape((B, S, self.num_heads, self.head_dim))
        k = self.k_proj(x).reshape((B, S, self.num_heads, self.head_dim))
        v = self.v_proj(x).reshape((B, S, self.num_heads, self.head_dim))

        if self.use_rope:
            cos = self.cos[:, :S, :, :]
            sin = self.sin[:, :S, :, :]
            q, k = apply_rotary_emb(q, k, cos, sin)

        # Fused XLA FlashAttention lowered directly to TPU systolic array
        attn_out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.reshape((B, S, D))
        return self.out_proj(attn_out)
