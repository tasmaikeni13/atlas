"""Production 125M Parameter Transformer Architecture with FlashAttention for Google Cloud TPUs."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn

from .flash_attention import FlashCausalAttention, precompute_rope_freqs


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (self.dim,), self.dtype)
        variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return (normed * scale).astype(x.dtype)


class TransformerMLP(nn.Module):
    """Feed-Forward Network with GeLU activation."""
    d_model: int
    d_ff: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.d_ff, use_bias=True, dtype=self.dtype)(x)
        h = nn.gelu(h)
        out = nn.Dense(self.d_model, use_bias=True, dtype=self.dtype)(h)
        return out


class TransformerBlock(nn.Module):
    """Single Transformer Decoder block with FlashAttention and Pre-RMSNorm."""
    d_model: int
    num_heads: int
    d_ff: int
    dtype: jnp.dtype = jnp.float32
    max_seq_len: int = 2048

    def setup(self):
        self.attn_norm = RMSNorm(self.d_model, dtype=self.dtype)
        self.attn = FlashCausalAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dtype=self.dtype,
            use_rope=True,
            max_seq_len=self.max_seq_len
        )
        self.mlp_norm = RMSNorm(self.d_model, dtype=self.dtype)
        self.mlp = TransformerMLP(d_model=self.d_model, d_ff=self.d_ff, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        h = x + self.attn(self.attn_norm(x), deterministic=deterministic)
        out = h + self.mlp(self.mlp_norm(h))
        return out


class Transformer125M(nn.Module):
    """125-Million Parameter Causal Transformer for Language Modeling on FineWeb-Edu.
    
    Configuration:
        vocab_size: 50,257 (GPT-2 BPE tokenizer vocabulary)
        d_model: 768
        num_layers: 12
        num_heads: 12 (head_dim = 64)
        d_ff: 3072
        max_seq_len: 1024
        Total Parameters: ~124.5M (with tied LM-head weights)
    """
    vocab_size: int = 50257
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 1024
    dtype: jnp.dtype = jnp.float32
    tie_embeddings: bool = True

    def setup(self):
        self.embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype
        )
        self.layers = [
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dtype=self.dtype,
                max_seq_len=self.max_seq_len,
                name=f"layer_{i}"
            )
            for i in range(self.num_layers)
        ]
        self.final_norm = RMSNorm(self.d_model, dtype=self.dtype)
        if not self.tie_embeddings:
            self.lm_head = nn.Dense(self.vocab_size, use_bias=False, dtype=self.dtype)

    def __call__(self, input_ids: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Autoregressive forward pass.
        
        Args:
            input_ids: (batch_size, seq_len) int32 tokens
        Returns:
            logits: (batch_size, seq_len, vocab_size) float32 next-token predictions
        """
        x = self.embed(input_ids)

        for layer in self.layers:
            x = layer(x, deterministic=deterministic)

        x = self.final_norm(x)

        if self.tie_embeddings:
            # Tied LM head: logits = x @ embed.weight.T
            embed_w = self.embed.variables["params"]["embedding"]
            logits = jnp.matmul(x, embed_w.T)
        else:
            logits = self.lm_head(x)

        return logits.astype(jnp.float32)
