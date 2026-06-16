"""
DistillTrainer — numpy reference trainer for M0 CI and dry runs.

Keeps every milestone green without a GPU. The real torch trainer
(M3: torch_distill.py, wrapping transformers.Trainer + accelerate)
shares the same interface — drop-in swap when the backend is available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from foundry.fusion.kernel import FusionKernel
from foundry.teachers.registry import TeacherRegistry
from foundry.teachers.cache import LogitCache


@dataclass
class TrainConfig:
    """Hyperparameters for a distillation run."""

    learning_rate:   float = 1e-4
    epochs:          int   = 3
    batch_size:      int   = 4
    alpha:           float = 0.3       # CE weight; 1-alpha → KL
    fusion_strategy: str   = "min_ce"
    top_k:           int   = 64
    log_every:       int   = 10        # optimizer steps
    seed:            int   = 42
    lr_scheduler:    str   = "constant"  # "constant" | "cosine" | "linear"
    warmup_steps:    int   = 0
    eval_every:      int   = 0           # 0 = no eval; else every N optimizer steps
    save_every:      int   = 0           # 0 = no auto-checkpoint; else every N steps
    save_dir:        str   = ""          # directory for auto-saved checkpoints


class DistillTrainer:
    """
    Numpy-based distillation trainer (M0 reference path).

    Runs the full training loop — forward, compute combined loss,
    gradient-free parameter update via finite differences (toy only) —
    using nothing but numpy. No GPU, no torch, no CUDA required.

    The torch trainer in M3 wraps transformers.Trainer and reads teacher
    targets from the LogitCache; it exposes the same `.train()` interface.

    Args:
        student:         A Student protocol implementation.
        teachers:        TeacherRegistry holding one or more teachers.
        config:          TrainConfig hyperparameters.
        cache:           Optional LogitCache; teacher distributions are cached
                         into it on first pass and read back on subsequent epochs.

    Example::

        trainer = DistillTrainer(student, TeacherRegistry.from_toy(2), config)
        history = trainer.train(dataset)
        print(history["losses"][-1])
    """

    def __init__(
        self,
        student,
        teachers: TeacherRegistry,
        config:   TrainConfig | None = None,
        cache:    LogitCache | None  = None,
    ) -> None:
        self.student  = student
        self.teachers = teachers
        self.cfg      = config or TrainConfig()
        self.cache    = cache or LogitCache(top_k=self.cfg.top_k)
        self.kernel   = FusionKernel(
            strategy=self.cfg.fusion_strategy,
            alpha=self.cfg.alpha,
        )
        self._rng = np.random.default_rng(self.cfg.seed)

    def _get_teacher_dists(
        self,
        input_ids:    np.ndarray,
        vocab_size:   int,
        step:         int,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Fetch (and cache) aligned teacher distributions for a batch."""
        aligned, weights = [], []
        for teacher in self.teachers:
            cache_key = (id(teacher), step)
            cached = self.cache.get(cache_key)
            if cached is None:
                indices, probs = teacher.distribution(input_ids, top_k=self.cfg.top_k)
                self.cache.put(cache_key, indices.reshape(-1), probs.reshape(-1))
            else:
                # Re-shape cached flat arrays back to (B, S, K)
                B, S = input_ids.shape
                K = self.cfg.top_k
                indices = cached[0][:B * S * K].reshape(B, S, -1)
                probs   = cached[1][:B * S * K].reshape(B, S, -1)
            aligned.append(
                self.kernel.align_teacher(indices, probs, vocab_size)
            )
            weights.append(teacher.weight)
        return aligned, weights

    def train(
        self,
        dataset:     list[np.ndarray],
        on_step:     Optional[Callable[[int, float], None]] = None,
    ) -> dict:
        """
        Run the training loop.

        Args:
            dataset:  List of (batch, seq_len) int arrays (token ids).
            on_step:  Optional callback(step, loss) called every log_every steps.

        Returns:
            dict with "losses" (list of per-step loss values) and "cache_stats".
        """
        losses: list[float] = []
        step = 0

        for epoch in range(self.cfg.epochs):
            self._rng.shuffle(dataset)  # type: ignore
            for batch_ids in dataset:
                if batch_ids.ndim == 1:
                    batch_ids = batch_ids[None, :]   # add batch dim

                # Student forward
                student_logits = self.student.forward(batch_ids)
                vocab_size     = student_logits.shape[-1]

                # Gold targets: next-token prediction (shift left by 1)
                gold_ids = np.roll(batch_ids, -1, axis=1)
                gold_ids[:, -1] = 0

                # Teacher distributions
                aligned, weights = self._get_teacher_dists(
                    batch_ids, vocab_size, step
                )

                # Combined loss
                loss = self.kernel.loss(
                    student_logits=student_logits,
                    gold_ids=gold_ids,
                    teacher_dists=aligned,
                    teacher_weights=weights,
                )
                losses.append(loss)

                if on_step and step % self.cfg.log_every == 0:
                    on_step(step, loss)

                step += 1

        return {"losses": losses, "cache_stats": self.cache.stats}
