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

    Args:
        name:       HF model ID, local path, or ``"org/model@rev"`` string.
        weight:     Relative contribution when fusing multiple teachers.
        ref:        Pre-parsed ``ModelRef`` (optional; parsed from name if omitted).
        model_type: ``"causal_lm"`` (default) for GPT/Llama-style teachers;
                    ``"encoder"`` for BERT/DeBERTa/RoBERTa-style teachers used
                    in embedding distillation.
    """

    def __init__(
        self,
        name:       str,
        weight:     float = 1.0,
        ref=None,
        model_type: str   = "causal_lm",
    ) -> None:
        self.name       = name
        self.weight     = weight
        self._ref       = ref
        self.model_type = model_type
        self._model     = None
        self._tok       = None

    @property
    def tokenizer(self):
        return self._tok

    def load(self, device: str = "auto") -> "HFTeacher":
        """
        Load model and tokenizer. Moves model to the appropriate device.
        Safe to call multiple times — no-op after the first load.

        Args:
            device: "auto", "cuda", "cpu", "mps", or a specific device string.

        Returns:
            self (for chaining: ``teacher.load().distribution(...)``)
        """
        if self._model is not None:
            return self
        from foundry.io import ModelRef, load_tokenizer
        from foundry.io.loader import load_model

        ref = self._ref or ModelRef.parse(self.name)

        if self.model_type == "encoder":
            from transformers import AutoModel
            model_cls = AutoModel
        else:
            from transformers import AutoModelForCausalLM
            model_cls = AutoModelForCausalLM

        self._model = load_model(ref, model_class=model_cls)
        self._tok   = load_tokenizer(ref)
        self._model.eval()
        self._device = self._resolve_device(device)
        self._model.to(self._device)
        return self

    @staticmethod
    def _resolve_device(device: str):
        import torch
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def distribution(
        self,
        input_ids: np.ndarray,
        top_k:     int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run a forward pass and return top-k logits as (indices, probs).

        Only valid for causal_lm teachers. Raises TypeError for encoders — use
        get_embeddings() instead.

        Args:
            input_ids: (batch, seq_len) int32 numpy array.
            top_k:     Number of top logits to return per position.

        Returns:
            indices: (batch, seq_len, top_k) int32
            probs:   (batch, seq_len, top_k) float32, sum ≈ 1 over last dim
        """
        if self.model_type == "encoder":
            raise TypeError(
                f"HFTeacher '{self.name}' is an encoder (model_type='encoder'). "
                "Encoders do not produce token logits. "
                "Use .get_embeddings(input_ids, attention_mask) to get pooled vectors."
            )
        if self._model is None:
            raise RuntimeError(
                f"HFTeacher '{self.name}' not loaded. Call .load() first."
            )
        import torch
        device = getattr(self, "_device", torch.device("cpu"))
        with torch.no_grad():
            ids_t  = torch.tensor(input_ids, dtype=torch.long, device=device)
            out    = self._model(input_ids=ids_t)
            logits = out.logits.float()                          # (B, S, V)
            probs  = torch.softmax(logits, dim=-1)
            k      = min(top_k, probs.shape[-1])
            top    = torch.topk(probs, k=k, dim=-1)
        return (
            top.indices.cpu().numpy().astype(np.int32),
            top.values.cpu().numpy().astype(np.float32),
        )

    def get_embeddings(
        self,
        input_ids:      np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        pool:           str = "mean",
    ) -> np.ndarray:
        """
        Get pooled sentence embeddings from an encoder teacher.

        Only valid for encoder teachers (model_type='encoder'). Raises TypeError
        for causal_lm teachers — use distribution() instead.

        Args:
            input_ids:      (batch, seq_len) int64 numpy array.
            attention_mask: (batch, seq_len) int64; ones if omitted.
            pool:           "mean" (default) or "cls".

        Returns:
            embeddings: (batch, hidden_size) float32 numpy array.
        """
        if self.model_type != "encoder":
            raise TypeError(
                f"HFTeacher '{self.name}' is a causal_lm teacher. "
                "Use .distribution() to get token-level logits."
            )
        if self._model is None:
            raise RuntimeError(
                f"HFTeacher '{self.name}' not loaded. Call .load() first."
            )
        import torch
        device = getattr(self, "_device", torch.device("cpu"))
        with torch.no_grad():
            ids_t  = torch.tensor(input_ids, dtype=torch.long, device=device)
            if attention_mask is not None:
                mask_t = torch.tensor(attention_mask, dtype=torch.long, device=device)
            else:
                mask_t = torch.ones_like(ids_t)
            out            = self._model(input_ids=ids_t, attention_mask=mask_t)
            hidden         = out.last_hidden_state.float()   # (B, S, H)
            if pool == "cls":
                emb = hidden[:, 0, :]
            else:  # mean pool over non-padding tokens
                mask_f = mask_t.unsqueeze(-1).float()        # (B, S, 1)
                emb    = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)
        return emb.cpu().numpy().astype(np.float32)


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
        """
        Build registry from HF model names. Models are NOT loaded here —
        call ``.load_all()`` or load each teacher individually.
        """
        w = weights or [1.0] * len(names)
        return cls([HFTeacher(name, weight=wt) for name, wt in zip(names, w)])

    def load_all(self, device: str = "auto") -> "TeacherRegistry":
        """
        Load all teachers. Logs progress to stdout.
        Skips ToyTeachers (they have no .load()).
        """
        for teacher in self._teachers:
            if hasattr(teacher, "load") and callable(teacher.load):
                print(f"  Loading teacher: {teacher.name} …")
                teacher.load(device=device)
        return self

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
