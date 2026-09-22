"""High-performance data loaders for Vision Transformers (CIFAR-10) and Causal Transformers (WikiText)."""

from __future__ import annotations

import os
import glob
from typing import Generator, List, Optional, Tuple
import numpy as np
from PIL import Image

def load_cifar10(
    data_dir: Optional[str] = None,
    max_train: int = 5000,
    max_test: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads CIFAR-10 images and labels normalized to [0, 1].
    
    Returns:
        (train_images, train_labels, test_images, test_labels)
        Images: (N, 32, 32, 3) float32
        Labels: (N,) int32
    """
    if data_dir is None:
        cand = os.path.join(os.getcwd(), "data/cifar10")
        data_dir = cand if os.path.exists(cand) else "data/cifar10"
    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def read_split(split_name: str, max_count: int) -> Tuple[np.ndarray, np.ndarray]:
        img_list = []
        lbl_list = []
        split_dir = os.path.join(data_dir, split_name)
        per_class = max_count // len(classes)
        
        for c in classes:
            c_dir = os.path.join(split_dir, c)
            files = sorted(glob.glob(os.path.join(c_dir, "*.png")))[:per_class]
            for f in files:
                with Image.open(f) as im:
                    arr = np.asarray(im, dtype=np.float32) / 255.0
                    img_list.append(arr)
                    lbl_list.append(class_to_idx[c])

        images = np.stack(img_list, axis=0)
        labels = np.array(lbl_list, dtype=np.int32)
        # Random shuffle
        idx = np.random.RandomState(42).permutation(len(labels))
        return images[idx], labels[idx]

    train_x, train_y = read_split("train", max_train)
    test_x, test_y = read_split("test", max_test)
    return train_x, train_y, test_x, test_y


def load_wikitext(
    filepath: str = "/home/tasma/algebraic-intelligence/data/wikitext-103-raw/wiki.valid.raw",
    vocab_size: int = 5000,
    seq_len: int = 64,
    max_tokens: int = 100000
) -> Tuple[np.ndarray, dict]:
    """Loads and tokenizes text corpus for Causal Language Transformer modeling."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()[:max_tokens * 6]

    # Simple byte-pair or character-level / word frequency tokenizer
    words = text.split()[:max_tokens]
    from collections import Counter
    counts = Counter(words)
    vocab = ["<pad>", "<unk>", "<bos>", "<eos>"] + [w for w, _ in counts.most_common(vocab_size - 4)]
    word2idx = {w: i for i, w in enumerate(vocab)}

    tokens = np.array([word2idx.get(w, 1) for w in words], dtype=np.int32)
    
    # Chunk into sequences of length seq_len
    num_seqs = len(tokens) // seq_len
    tokens = tokens[: num_seqs * seq_len].reshape(num_seqs, seq_len)
    return tokens, word2idx


class BatchIterator:
    """Fast in-memory batch iterator for JAX arrays."""

    def __init__(self, data: Tuple[np.ndarray, ...], batch_size: int, shuffle: bool = True, seed: int = 42):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data[0])
        self.rng = np.random.RandomState(seed)

    def __iter__(self) -> Generator[Tuple[np.ndarray, ...], None, None]:
        indices = np.arange(self.num_samples)
        if self.shuffle:
            indices = self.rng.permutation(indices)
        for i in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            if len(batch_idx) == self.batch_size:
                yield tuple(d[batch_idx] for d in self.data)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size
