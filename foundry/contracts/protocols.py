from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class ArchConfig:
    """Architecture dimensions — the single source of truth for size arithmetic."""

    n_layers: int
    d_model: int
    vocab_size: int
    n_heads: int = 0
    d_ff: int = 0            # defaults to 4 × d_model if 0
    tie_embeddings: bool = True
    name: str = "unnamed"

    def __post_init__(self) -> None:
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model

    def params_estimate(self) -> int:
        """
        Rough parameter count using the ~12·d_model²·n_layers rule.
        Embedding params added separately; shared (tied) embeddings counted once.
        """
        attn = 4 * self.d_model * self.d_model   # Q K V O projections
        ff   = 2 * self.d_model * self.d_ff       # two FF matrices
        per_layer = attn + ff
        embed = self.vocab_size * self.d_model
        total = self.n_layers * per_layer + embed
        if not self.tie_embeddings:
            total += embed    # separate lm_head
        return total

    def params_b(self) -> float:
        """Billions of parameters, rounded to 1 dp."""
        return round(self.params_estimate() / 1e9, 1)

    def flops_per_token(self) -> int:
        """Approx training FLOPs per token: 6·N (Chinchilla rule)."""
        return 6 * self.params_estimate()

    def training_cost_estimate(
        self,
        tokens: float,
        h100_tflops: float = 4e14,
        utilization: float = 0.40,
        usd_per_h100_hr: float = 2.50,
    ) -> dict:
        """
        Rough cost estimate for a training run.

        Args:
            tokens: Training tokens (e.g. 1e12 for 1T).
            h100_tflops: Theoretical peak H100 TFLOPs/s (default 4e14).
            utilization: MFU fraction (default 40%).
            usd_per_h100_hr: On-demand H100 price.

        Returns:
            dict with flops, gpu_hours, usd_estimate.
        """
        total_flops = self.flops_per_token() * tokens
        effective_tflops = h100_tflops * utilization
        gpu_seconds = total_flops / effective_tflops
        gpu_hours   = gpu_seconds / 3600
        usd = gpu_hours * usd_per_h100_hr
        return {
            "total_flops":  total_flops,
            "gpu_hours":    round(gpu_hours, 1),
            "usd_estimate": round(usd, 0),
        }

    def shape_warning(self) -> Optional[str]:
        """Warn when the model is deep/narrow (depth-scaling trap).

        At d_model=4096 a normal model has ~32 layers (d_model/128).
        A ratio above 2.0 means the model is being grown too deep for its width.
        """
        expected_layers = self.d_model / 128
        ratio = self.n_layers / max(1, expected_layers)
        if ratio > 2.0:
            return (
                f"Deep/narrow shape detected (layers={self.n_layers}, "
                f"d_model={self.d_model}). "
                "Consider widening the seed — depth scales linearly but "
                "width scales quadratically in parameter count."
            )
        return None


@runtime_checkable
class Student(Protocol):
    """
    Minimal interface every student model must satisfy.
    Implement this for a custom architecture; the factory does the rest.
    """

    config: ArchConfig

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Return logits of shape (batch, seq_len, vocab_size)."""
        ...

    @property
    def hidden_states(self) -> Optional[np.ndarray]:
        """Optional: last forward pass hidden states for activation distillation."""
        return None

    @property
    def tokenizer(self) -> object:
        """The tokenizer attached to this student."""
        ...

    def parameters(self) -> list:
        """Return trainable parameter arrays (numpy or torch tensors)."""
        ...


@runtime_checkable
class Teacher(Protocol):
    """
    Interface every teacher model must expose to the fusion kernel.
    HFTeacher is the standard implementation; ToyTeacher is used in tests.
    """

    name: str
    weight: float

    @property
    def tokenizer(self) -> object:
        """The teacher's tokenizer (may differ from the student's)."""
        ...

    def distribution(
        self,
        input_ids: np.ndarray,
        top_k: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (indices, probs) of shape (batch, seq_len, top_k).
        Indices are in the teacher's vocab space.
        """
        ...


@runtime_checkable
class TokenizerAlignment(Protocol):
    """
    Maps a teacher's token distribution into the student's vocab space.
    Each alignment strategy implements this; the kernel is strategy-agnostic.
    """

    def map(
        self,
        teacher_indices: np.ndarray,
        teacher_probs:   np.ndarray,
        student_vocab_size: int,
    ) -> np.ndarray:
        """
        Return a dense probability array of shape (batch, seq_len, student_vocab_size).
        Missing positions should be zero (not renormalised here — kernel handles that).
        """
        ...
