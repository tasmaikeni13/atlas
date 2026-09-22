"""Hardware-accelerated Vision Transformer (ViT) on Google Cloud TPUs for ImageNet-100."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp


class FusedMultiHeadAttention(nn.Module):
    """TPU-optimized Multi-Head Attention using XLA fused dot-product attention."""
    num_heads: int
    head_dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape
        proj_dim = self.num_heads * self.head_dim

        # Fused Q, K, V projection
        qkv = nn.Dense(3 * proj_dim, use_bias=True, dtype=self.dtype, name="qkv")(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # XLA lowers jax.nn.dot_product_attention directly into TPU systolic MXU kernels
        # Expected shape: (batch, seq_len, num_heads, head_dim)
        out = jax.nn.dot_product_attention(
            query=q,
            key=k,
            value=v,
            is_causal=False  # Bidirectional self-attention for Vision Transformers
        )
        # out: (batch, seq_len, num_heads, head_dim)
        out = out.reshape(batch_size, seq_len, proj_dim)
        return nn.Dense(embed_dim, use_bias=True, dtype=self.dtype, name="proj")(out)


class MLPBlock(nn.Module):
    """Feed-forward network with GELU activation."""
    hidden_dim: int
    out_dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, dtype=self.dtype, name="fc1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, dtype=self.dtype, name="fc2")(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer block."""
    num_heads: int
    head_dim: int
    mlp_dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Attention sub-layer with residual
        norm_x = nn.LayerNorm(dtype=self.dtype, name="norm1")(x)
        attn_out = FusedMultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            name="attn"
        )(norm_x, deterministic=deterministic)
        x = x + attn_out

        # MLP sub-layer with residual
        norm_x2 = nn.LayerNorm(dtype=self.dtype, name="norm2")(x)
        mlp_out = MLPBlock(
            hidden_dim=self.mlp_dim,
            out_dim=x.shape[-1],
            dtype=self.dtype,
            name="mlp"
        )(norm_x2)
        return x + mlp_out


class VisionTransformerImageNet(nn.Module):
    """Vision Transformer (ViT) architecture configured for ImageNet-100 classification.
    
    Supports ViT-Small/16, ViT-Base/16, or custom configurations.
    Default: ViT-Small/16 (d_model=384, 12 layers, 6 heads, patch_size=16x16, ~21.7M params).
    """
    num_classes: int = 100
    img_size: int = 224
    patch_size: int = 16
    d_model: int = 384
    num_layers: int = 12
    num_heads: int = 6
    mlp_dim: int = 1536
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # x: (batch, H, W, C) where C = 3
        batch_size, H, W, C = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, f"Image dimensions ({H}, {W}) must be divisible by patch size {p}"
        num_patches = (H // p) * (W // p)

        # Patch projection: conv2d with kernel=patch_size and stride=patch_size
        patches = nn.Conv(
            features=self.d_model,
            kernel_size=(p, p),
            strides=(p, p),
            padding="VALID",
            dtype=self.dtype,
            name="patch_embed"
        )(x)
        # patches: (batch, num_patches_h, num_patches_w, d_model) -> (batch, num_patches, d_model)
        patches = patches.reshape(batch_size, num_patches, self.d_model)

        # Learnable CLS token
        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.d_model),
            self.dtype
        )
        cls_tokens = jnp.broadcast_to(cls_token, (batch_size, 1, self.d_model))

        # Concatenate CLS token with patch tokens: (batch, 1 + num_patches, d_model)
        tokens = jnp.concatenate([cls_tokens, patches], axis=1)
        seq_len = tokens.shape[1]

        # Learnable 1D position embeddings
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, seq_len, self.d_model),
            self.dtype
        )
        tokens = tokens + pos_embed

        # Transformer encoder stack
        head_dim = self.d_model // self.num_heads
        for i in range(self.num_layers):
            tokens = TransformerEncoderBlock(
                num_heads=self.num_heads,
                head_dim=head_dim,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype,
                name=f"block_{i}"
            )(tokens, deterministic=deterministic)

        # Final pre-classifier LayerNorm
        tokens = nn.LayerNorm(dtype=self.dtype, name="norm_final")(tokens)

        # Extract CLS token representation for classification
        cls_rep = tokens[:, 0]

        # Linear classification head to ImageNet-100 logits
        logits = nn.Dense(
            features=self.num_classes,
            dtype=self.dtype,
            name="head"
        )(cls_rep)

        return logits.astype(jnp.float32)


def create_vit_small_imagenet(num_classes: int = 100, img_size: int = 224, patch_size: int = 16) -> VisionTransformerImageNet:
    """ViT-Small/16 configuration (21.7M parameters)."""
    return VisionTransformerImageNet(
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        d_model=384,
        num_layers=12,
        num_heads=6,
        mlp_dim=1536,
        dtype=jnp.bfloat16
    )


def create_vit_tiny_imagenet(num_classes: int = 100, img_size: int = 224, patch_size: int = 16) -> VisionTransformerImageNet:
    """ViT-Tiny/16 configuration (5.7M parameters) for fast hyperparameter sweeps."""
    return VisionTransformerImageNet(
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        d_model=192,
        num_layers=12,
        num_heads=3,
        mlp_dim=768,
        dtype=jnp.bfloat16
    )


# Alias for unified naming convention
VisionTransformer = VisionTransformerImageNet

