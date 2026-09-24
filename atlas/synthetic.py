"""Small deterministic tasks for offline diagnostic smoke tests."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


class ColorPatternDataset:
    """Balanced binary images with a recoverable channel-mean label signal."""

    def __init__(
        self,
        img_size: int = 64,
        seed: int = 42,
        signal: float = 0.08,
        noise_std: float = 0.5,
    ):
        if img_size <= 0 or signal <= 0 or noise_std <= 0:
            raise ValueError("image size, signal, and noise must be positive")
        self.img_size = img_size
        self.signal = signal
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def generate_batch(self, batch_size: int):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("batch size must be even and at least two")
        labels = np.tile(np.array([0, 1], dtype=np.int32), batch_size // 2)
        self.rng.shuffle(labels)
        images = self.rng.normal(
            0.0, self.noise_std,
            size=(batch_size, self.img_size, self.img_size, 3),
        ).astype(np.float32)
        images[labels == 0, :, :, 0] += self.signal
        images[labels == 1, :, :, 2] += self.signal
        return jnp.asarray(images), jnp.asarray(labels)
