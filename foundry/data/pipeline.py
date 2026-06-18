"""
DataPipeline — unified data loader for foundry trainers.

Converts heterogeneous data sources into the batched numpy arrays that
TorchDistillTrainer, CachedDistillTrainer, and EmbeddingDistillTrainer expect.

Accepted source types
---------------------
• HF ``datasets.Dataset`` (finite, indexable)
• HF ``datasets.IterableDataset`` (streaming, no RAM limit)
• ``list[str]``        — tokenized on the fly (requires tokenizer)
• ``list[dict]``       — with ``input_ids`` / ``text`` key
• ``list[np.ndarray]`` — already tokenized rows, one per example

Modes
-----
• ``"lm"``    → yields ``(B, S)`` int32 arrays   (TorchDistillTrainer)
• ``"embed"`` → yields ``{"input_ids": ..., "attention_mask": ...}`` dicts
                (EmbeddingDistillTrainer)

Quick start
-----------
::

    from foundry.data import DataPipeline
    from transformers import AutoTokenizer

    tok  = AutoTokenizer.from_pretrained("gpt2")
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)

    pipe = DataPipeline(data, tokenizer=tok, batch_size=8, max_length=512, mode="lm")
    result = trainer.train(pipe)

Streaming shuffle::

    pipe = DataPipeline(data, tokenizer=tok, batch_size=8,
                        shuffle_buffer=1000)   # reservoir shuffle on-the-fly
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, Iterator, Optional

import numpy as np


class DataPipeline:
    """
    Converts a data source into batched numpy arrays for foundry trainers.

    Args:
        source:         Any iterable of examples. Accepts HF Dataset /
                        IterableDataset, list of strings, list of dicts,
                        or list of numpy arrays.
        tokenizer:      HF tokenizer or any callable ``str → list[int]``.
                        Required when the source contains raw strings.
        batch_size:     Examples per yielded batch.
        max_length:     Truncate sequences at this length; pad shorter ones.
        mode:           ``"lm"`` (int array) or ``"embed"`` (dict with mask).
        shuffle_buffer: Reservoir buffer size for streaming shuffle.
                        0 = no shuffle. For finite datasets, shuffle the whole
                        list instead (no buffer needed).
        text_column:    Column name for raw text in HF dataset rows.
        ids_column:     Column name for pre-tokenised ids in HF dataset rows.
        mask_column:    Column name for attention mask in HF dataset rows.
        pad_id:         Token id used for padding. Default 0.
        drop_last:      Discard the final partial batch. Default False.
    """

    def __init__(
        self,
        source,
        tokenizer:      Optional[Callable] = None,
        batch_size:     int  = 8,
        max_length:     int  = 512,
        mode:           str  = "lm",
        shuffle_buffer: int  = 0,
        text_column:    str  = "text",
        ids_column:     str  = "input_ids",
        mask_column:    str  = "attention_mask",
        pad_id:         int  = 0,
        drop_last:      bool = False,
        label_column:   Optional[str] = None,
        label_pad_id:   int  = -100,
    ) -> None:
        self.source         = source
        self.tokenizer      = tokenizer
        self.batch_size     = batch_size
        self.max_length     = max_length
        self.mode           = mode
        self.shuffle_buffer = shuffle_buffer
        self.text_column    = text_column
        self.ids_column     = ids_column
        self.mask_column    = mask_column
        self.pad_id         = pad_id
        self.drop_last      = drop_last
        self.label_column   = label_column
        self.label_pad_id   = label_pad_id

    # ── Length ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """
        Number of batches.

        Raises ``TypeError`` for streaming sources with unknown length.
        """
        try:
            n = len(self.source)  # type: ignore[arg-type]
        except TypeError:
            raise TypeError(
                "DataPipeline source has unknown length (streaming). "
                "Pass total_steps= to trainer.train() explicitly."
            )
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    # ── Iteration ────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator:
        """Yield batches from the source."""
        src = self._shuffled_iter() if self.shuffle_buffer > 0 else iter(self.source)

        buf_ids:  list[np.ndarray] = []
        buf_mask: list[np.ndarray] = []
        buf_lab:  list = []

        for example in src:
            ids_arr, mask_arr = self._to_ids_and_mask(example)
            buf_ids.append(ids_arr)
            buf_mask.append(mask_arr)
            if self.label_column is not None:
                buf_lab.append(self._extract_label(example))

            if len(buf_ids) == self.batch_size:
                yield self._make_batch(buf_ids, buf_mask, buf_lab)
                buf_ids, buf_mask, buf_lab = [], [], []

        if buf_ids and not self.drop_last:
            yield self._make_batch(buf_ids, buf_mask, buf_lab)

    def _extract_label(self, example):
        """Return a scalar label, or a token-label sequence padded to max_length."""
        if not isinstance(example, dict) or self.label_column not in example:
            raise ValueError(
                f"label_column='{self.label_column}' not found in example. "
                f"Each row must be a dict containing that key."
            )
        lab = example[self.label_column]
        if isinstance(lab, (list, tuple, np.ndarray)):     # token-level labels
            lab = list(lab)[:self.max_length]
            arr = np.full(self.max_length, self.label_pad_id, dtype=np.int64)
            arr[:len(lab)] = lab
            return arr
        return int(lab)                                    # sequence-level label

    # ── Conversion helpers ───────────────────────────────────────────────────

    def _to_ids_and_mask(self, example) -> tuple[np.ndarray, np.ndarray]:
        """Convert one example (any type) to (ids, mask) numpy arrays."""
        if isinstance(example, str):
            return self._pad(self._tokenize(example))

        if isinstance(example, dict):
            # Pre-tokenised dict with "input_ids" key
            if self.ids_column in example:
                ids  = list(example[self.ids_column])[:self.max_length]
                if self.mask_column in example:
                    mask = list(example[self.mask_column])[:self.max_length]
                    return self._pad_with_mask(ids, mask)
                return self._pad(ids)
            # Dict with raw text key
            if self.text_column in example:
                return self._pad(self._tokenize(str(example[self.text_column])))
            raise ValueError(
                f"dict example has neither '{self.ids_column}' nor '{self.text_column}' key. "
                f"Available keys: {list(example.keys())}"
            )

        if isinstance(example, np.ndarray):
            if example.ndim == 1:
                return self._pad(example.tolist())
            # 2-D row — take first row
            return self._pad(example[0].tolist())

        if isinstance(example, (list, tuple)):
            return self._pad(list(example)[:self.max_length])

        raise TypeError(
            f"DataPipeline cannot convert {type(example).__name__} to token ids. "
            "Provide str, dict, numpy array, or list."
        )

    def _tokenize(self, text: str) -> list[int]:
        if self.tokenizer is None:
            raise ValueError(
                "DataPipeline.tokenizer is required for string input. "
                "Pass tokenizer= when creating the pipeline."
            )
        result = self.tokenizer(text)
        if isinstance(result, dict):
            ids = result.get("input_ids", [])
        elif isinstance(result, (list, np.ndarray)):
            ids = list(result)
        else:
            # HF tokenizer returns a dict-like BatchEncoding
            try:
                ids = list(result["input_ids"])
            except (TypeError, KeyError):
                ids = list(result)
        return ids[:self.max_length]

    def _pad(self, ids: list) -> tuple[np.ndarray, np.ndarray]:
        L        = min(len(ids), self.max_length)
        ids_arr  = np.full(self.max_length, self.pad_id, dtype=np.int32)
        mask_arr = np.zeros(self.max_length,              dtype=np.int32)
        ids_arr[:L]  = ids[:L]
        mask_arr[:L] = 1
        return ids_arr, mask_arr

    def _pad_with_mask(self, ids: list, mask: list) -> tuple[np.ndarray, np.ndarray]:
        L        = min(len(ids), self.max_length)
        ids_arr  = np.full(self.max_length, self.pad_id, dtype=np.int32)
        mask_arr = np.zeros(self.max_length,              dtype=np.int32)
        ids_arr[:L]  = ids[:L]
        mask_arr[:L] = mask[:L]
        return ids_arr, mask_arr

    def _make_batch(
        self,
        ids_list:  list[np.ndarray],
        mask_list: list[np.ndarray],
        lab_list:  list | None = None,
    ):
        ids  = np.stack(ids_list,  axis=0)   # (B, S)
        mask = np.stack(mask_list, axis=0)   # (B, S)
        # When labels are requested, always emit a dict (head trainers need them).
        if lab_list:
            labels = (
                np.stack(lab_list, axis=0)              # token labels → (B, S)
                if isinstance(lab_list[0], np.ndarray)
                else np.asarray(lab_list, dtype=np.int64)  # sequence labels → (B,)
            )
            return {"input_ids": ids, "attention_mask": mask, "labels": labels}
        if self.mode == "lm":
            return ids
        return {"input_ids": ids, "attention_mask": mask}

    # ── Streaming shuffle ────────────────────────────────────────────────────

    def _shuffled_iter(self) -> Iterator:
        """
        Reservoir-buffer shuffle for streaming sources.

        Maintains a fixed-size buffer; on each step, a random element is
        evicted and the new example inserted.  Finite sources shuffle the
        full list directly (buffer size does not limit randomness).
        """
        try:
            items = list(self.source)
            random.shuffle(items)
            yield from items
            return
        except Exception:
            pass   # fall through to buffer-based streaming shuffle

        buf = []
        for example in self.source:
            buf.append(example)
            if len(buf) >= self.shuffle_buffer:
                idx = random.randrange(len(buf))
                yield buf[idx]
                buf[idx] = buf[-1]
                buf.pop()
        random.shuffle(buf)
        yield from buf
