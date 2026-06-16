"""
Teacher registry — build and hold a pool of teacher models for distillation.

M0: ToyTeacher (deterministic random, no GPU) for CI.
M1: HFTeacher (loads real HF models, requires torch backend).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ToyTeacher:
    """
    Deterministic fake teacher for tests and dry-run demos.
    Returns a reproducible softmax distribution over a small vocab.
    """

    name:       str
    weight:     float = 1.0
    vocab_size: int   = 100
    top_k:      int   = 10
    seed:       int   = 42

    @property
    def tokenizer(self):
        return None   # identity tokenizer — indices are already in student space

    def distribution(
        self,
        input_ids: np.ndarray,
        top_k: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return deterministic top-k (indices, probs) for the given input."""
        rng  = np.random.default_rng(self.seed + int(input_ids.sum()))
        B, S = input_ids.shape
        k    = min(top_k, self.vocab_size)
        # Use integers (with replacement) — this is a toy teacher, uniqueness not required
        indices = rng.integers(0, self.vocab_size, size=(B, S, k))
        logits = rng.random((B, S, k)).astype(np.float32)
        probs  = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        return indices.astype(np.int32), probs


class HFTeacher:
    """
    Teacher backed by a real Hugging Face model.
    Lazy-loaded; model stays on CPU/GPU and is not moved between calls.

    M0 status: interface defined, load() is a stub that raises ImportError
    unless the torch backend is installed.
    """

    def __init__(self, name: str, weight: float = 1.0, ref=None) -> None:
        self.name   = name
        self.weight = weight
        self._ref   = ref
        self._model = None
        self._tok   = None

    @property
    def tokenizer(self):
        return self._tok

    def load(self) -> None:
        """Load model and tokenizer from the HF hub. Requires [torch] extra."""
        from foundry.io import ModelRef, load_model, load_tokenizer
        ref = self._ref or ModelRef.parse(self.name)
        self._model = load_model(ref)
        self._tok   = load_tokenizer(ref)

    def distribution(
        self,
        input_ids: np.ndarray,
        top_k: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run forward pass and return top-k (indices, probs)."""
        if self._model is None:
            raise RuntimeError(
                f"HFTeacher '{self.name}' not loaded. Call .load() first."
            )
        import torch
        with torch.no_grad():
            t = torch.from_numpy(input_ids)
            logits = self._model(t).logits.float()
            probs  = torch.softmax(logits, dim=-1)
            top    = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
        return top.indices.numpy().astype(np.int32), top.values.numpy()


class TeacherRegistry:
    """
    Pool of teachers for a distillation run.

    Usage::

        registry = TeacherRegistry.from_names(
            ["org/reasoning-teacher", "org/code-teacher"],
            weights=[1.0, 0.8],
        )
        for teacher in registry:
            indices, probs = teacher.distribution(batch_input_ids)
    """

    def __init__(self, teachers: list) -> None:
        self._teachers = teachers

    @classmethod
    def from_names(
        cls,
        names:   list[str],
        weights: list[float] | None = None,
    ) -> "TeacherRegistry":
        """Build registry from HF model names. Requires [torch] extra."""
        w = weights or [1.0] * len(names)
        return cls([HFTeacher(name, weight=wt) for name, wt in zip(names, w)])

    @classmethod
    def from_toy(
        cls,
        n: int = 2,
        vocab_size: int = 100,
        weights: list[float] | None = None,
    ) -> "TeacherRegistry":
        """Build a registry of ToyTeachers (no GPU, for tests)."""
        w = weights or [1.0] * n
        return cls([
            ToyTeacher(name=f"toy-teacher-{i}", weight=wt, vocab_size=vocab_size, seed=i * 7)
            for i, wt in enumerate(w)
        ])

    def __iter__(self):
        return iter(self._teachers)

    def __len__(self) -> int:
        return len(self._teachers)

    @property
    def weights(self) -> list[float]:
        return [t.weight for t in self._teachers]
