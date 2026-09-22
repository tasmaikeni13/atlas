"""FineWeb-Edu Data Pipeline for 1B Token Large-Scale Transformer Training."""

from __future__ import annotations

import os
import pathlib
import time
from typing import Generator, Iterator, Optional, Tuple
import numpy as np
import tiktoken


class FineWebEduDataset:
    """High-throughput FineWeb-Edu dataset loader and streamer for 1B tokens.
    
    Supports:
    1. Direct streaming from HuggingFace `HuggingFaceFW/fineweb-edu` (sample-10BT or default).
    2. Local memmapped `.bin` cache for zero-overhead disk streaming.
    3. Fast BPE tokenization via tiktoken (`gpt2` encoding).
    4. Deterministic synthetic token stream for fast offline smoke testing.
    """

    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        subset: str = "sample-10BT",
        cache_dir: str = "data/fineweb_edu",
        seq_len: int = 1024,
        vocab_size: int = 50257,
        synthetic_fallback: bool = False
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.cache_dir = pathlib.Path(cache_dir)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.synthetic_fallback = synthetic_fallback
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.tokenizer = tiktoken.get_encoding("gpt2")
        except Exception:
            self.tokenizer = None

    def get_stream(
        self,
        batch_size: int = 64,
        max_tokens: int = 1_000_000_000,
        split: str = "train",
        seed: int = 42
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yields (inputs, targets) batches of shape (batch_size, seq_len).
        
        Inputs: tokens[:, :-1] (or tokens[:, :seq_len])
        Targets: tokens[:, 1:] (or tokens[:, 1:seq_len+1])
        """
        # Check if local cached bin exists
        cache_file = self.cache_dir / f"{split}_{self.subset}.bin"
        if cache_file.exists() and not self.synthetic_fallback:
            yield from self._stream_from_bin(cache_file, batch_size, max_tokens)
            return

        if self.synthetic_fallback:
            yield from self._stream_synthetic(batch_size, max_tokens, seed)
            return

        try:
            import datasets
            print(f"[FineWebEdu] Connecting to Hugging Face stream: {self.dataset_name} ({self.subset})...")
            ds = datasets.load_dataset(
                self.dataset_name,
                name=self.subset,
                split=split,
                streaming=True
            )
            yield from self._stream_from_hf(ds, batch_size, max_tokens)
        except Exception as e:
            print(f"[FineWebEdu] Streaming failed ({e}), falling back to synthetic generator for smoke testing.")
            yield from self._stream_synthetic(batch_size, max_tokens, seed)

    def _stream_from_hf(
        self,
        ds,
        batch_size: int,
        max_tokens: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        token_buffer = []
        tokens_yielded = 0
        chunk_size = self.seq_len + 1  # 1 extra for causal target

        for sample in ds:
            text = sample.get("text", "")
            if not text.strip():
                continue
            tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            tokens.append(self.tokenizer.eot_token)
            token_buffer.extend(tokens)

            while len(token_buffer) >= batch_size * chunk_size:
                batch_tokens = token_buffer[: batch_size * chunk_size]
                token_buffer = token_buffer[batch_size * chunk_size :]

                arr = np.array(batch_tokens, dtype=np.int32).reshape(batch_size, chunk_size)
                inputs = arr[:, :-1]
                targets = arr[:, 1:]
                tokens_yielded += batch_size * self.seq_len
                yield inputs, targets

                if tokens_yielded >= max_tokens:
                    return

    def _stream_from_bin(
        self,
        cache_file: pathlib.Path,
        batch_size: int,
        max_tokens: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Memory-mapped streaming from pre-tokenized binary file."""
        mmap_arr = np.memmap(cache_file, dtype=np.int32, mode="r")
        total_available = len(mmap_arr)
        chunk_size = self.seq_len + 1
        tokens_per_batch = batch_size * chunk_size
        tokens_yielded = 0

        idx = 0
        while idx + tokens_per_batch <= total_available and tokens_yielded < max_tokens:
            batch_raw = mmap_arr[idx : idx + tokens_per_batch].reshape(batch_size, chunk_size)
            inputs = np.array(batch_raw[:, :-1], dtype=np.int32)
            targets = np.array(batch_raw[:, 1:], dtype=np.int32)
            idx += tokens_per_batch
            tokens_yielded += batch_size * self.seq_len
            yield inputs, targets

    def _stream_synthetic(
        self,
        batch_size: int,
        max_tokens: int,
        seed: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Fast synthetic token stream mirroring language distributions for offline testing."""
        rng = np.random.default_rng(seed)
        tokens_yielded = 0
        chunk_size = self.seq_len + 1

        # Zipfian-like token frequency distribution
        vocab_probs = 1.0 / (np.arange(1, self.vocab_size + 1) ** 0.8)
        vocab_probs /= vocab_probs.sum()

        while tokens_yielded < max_tokens:
            arr = rng.choice(self.vocab_size, size=(batch_size, chunk_size), p=vocab_probs).astype(np.int32)
            inputs = arr[:, :-1]
            targets = arr[:, 1:]
            tokens_yielded += batch_size * self.seq_len
            yield inputs, targets
