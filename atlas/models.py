"""Transformer architectures for Vision and NLP using pure attention mechanisms."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Dict
import jax
import jax.numpy as jnp
import flax.linen as nn

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism."""
    num_heads: int
    d_model: int
    dropout_rate: float = 0.0
    causal: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        head_dim = self.d_model // self.num_heads
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        q = nn.Dense(self.d_model, name="q_proj")(x)
        k = nn.Dense(self.d_model, name="k_proj")(x)
        v = nn.Dense(self.d_model, name="v_proj")(x)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim).swapaxes(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, head_dim).swapaxes(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, head_dim).swapaxes(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(head_dim)
        scores = jnp.matmul(q, k.swapaxes(-1, -2)) * scale

        if self.causal:
            mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            scores = jnp.where(mask[None, None, :, :], scores, -1e9)

        attn_weights = jax.nn.softmax(scores, axis=-1)
        if not deterministic and self.dropout_rate > 0.0:
            attn_weights = nn.Dropout(self.dropout_rate)(attn_weights, deterministic=deterministic)

        out = jnp.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        out = out.swapaxes(1, 2).reshape(batch_size, seq_len, self.d_model)
        out = nn.Dense(self.d_model, name="out_proj")(out)
        return out


class TransformerBlock(nn.Module):
    """Standard Pre-LN Transformer Layer."""
    num_heads: int
    d_model: int
    d_ff: int
    dropout_rate: float = 0.0
    causal: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Pre-LN Self-Attention
        norm1 = nn.LayerNorm(name="ln1")(x)
        attn = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            causal=self.causal,
            name="attn"
        )(norm1, deterministic=deterministic)
        x = x + attn

        # Pre-LN MLP
        norm2 = nn.LayerNorm(name="ln2")(x)
        mlp = nn.Dense(self.d_ff, name="mlp_dense1")(norm2)
        mlp = nn.gelu(mlp)
        mlp = nn.Dense(self.d_model, name="mlp_dense2")(mlp)
        if not deterministic and self.dropout_rate > 0.0:
            mlp = nn.Dropout(self.dropout_rate)(mlp, deterministic=deterministic)
        x = x + mlp
        return x


class VisionTransformer(nn.Module):
    """Pure Attention Vision Transformer (ViT).
    
    Processes 2D images as sequences of flattened image patches with pure self-attention.
    Zero convolutions.
    """
    patch_size: int = 4
    num_classes: int = 10
    d_model: int = 128
    d_ff: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # x shape: (B, H, W, C)
        B, H, W, C = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch_size"
        num_patches = (H // p) * (W // p)
        patch_dim = p * p * C

        # Reshape into patches: (B, num_patches, patch_dim)
        patches = x.reshape(B, H // p, p, W // p, p, C)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(B, num_patches, patch_dim)

        # Patch projection to d_model
        x_proj = nn.Dense(self.d_model, name="patch_embed")(patches)

        # Class token
        cls_token = self.param("cls_token", nn.initializers.normal(stddev=0.02), (1, 1, self.d_model))
        cls_broadcast = jnp.broadcast_to(cls_token, (B, 1, self.d_model))
        tokens = jnp.concatenate([cls_broadcast, x_proj], axis=1)  # (B, num_patches + 1, d_model)

        # Position embeddings
        pos_embed = self.param("pos_embed", nn.initializers.normal(stddev=0.02), (1, num_patches + 1, self.d_model))
        tokens = tokens + pos_embed

        # Transformer Encoder Blocks
        for i in range(self.num_layers):
            tokens = TransformerBlock(
                num_heads=self.num_heads,
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                causal=False,
                name=f"block_{i}"
            )(tokens, deterministic=deterministic)

        # LayerNorm and Classification Head on CLS token
        tokens = nn.LayerNorm(name="final_ln")(tokens)
        cls_out = tokens[:, 0, :]
        logits = nn.Dense(self.num_classes, name="head")(cls_out)
        return logits


class CausalTransformer(nn.Module):
    """Causal Autoregressive Language Transformer (GPT-style).
    
    Pure causal self-attention for natural language sequence modeling.
    """
    vocab_size: int = 10000
    max_seq_len: int = 128
    d_model: int = 128
    d_ff: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # tokens shape: (B, seq_len)
        B, seq_len = tokens.shape
        assert seq_len <= self.max_seq_len, f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"

        token_embed = nn.Embed(self.vocab_size, self.d_model, name="token_embed")(tokens)
        
        pos_indices = jnp.arange(seq_len)[None, :]
        pos_embed = nn.Embed(self.max_seq_len, self.d_model, name="pos_embed")(pos_indices)
        
        x = token_embed + pos_embed

        for i in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                causal=True,
                name=f"block_{i}"
            )(x, deterministic=deterministic)

        x = nn.LayerNorm(name="final_ln")(x)
        logits = nn.Dense(self.vocab_size, name="lm_head")(x)
        return logits
