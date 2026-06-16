"""
TorchDistillTrainer — single-GPU distillation trainer.

Full production feature set:
  • Mixed precision (torch_dtype = "bfloat16" | "float16")
  • Gradient accumulation
  • LR scheduler: "constant" | "cosine" | "linear" with warmup
  • Reproducible seed (torch + numpy + random)
  • Dataset shuffling
  • Eval loop (eval_every, returns eval_losses per step)
  • Auto-checkpoint every N steps (save_every, save_dir)
  • Checkpoint save / resume
  • OOM error with actionable message
  • Optional W&B / TensorBoard logging
  • Accepts any iterable dataset (list, DataPipeline, HF IterableDataset)
"""
from __future__ import annotations

import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from foundry.training.distill import TrainConfig
from foundry.training._logger import _FoundryLogger
from foundry.training._scheduler import build_scheduler


@dataclass
class TorchTrainConfig(TrainConfig):
    """Extends TrainConfig with torch-specific params."""

    max_grad_norm:           float = 1.0
    weight_decay:            float = 0.01
    device:                  str   = "auto"
    grad_accumulation_steps: int   = 1
    torch_dtype:             str   = "float32"    # "float32" | "bfloat16" | "float16"
    log_backend:             str   = "none"        # "none" | "wandb" | "tensorboard"
    run_name:                str   = ""
    project:                 str   = "olaverse-foundry"
    log_loss_every:          int   = 0             # deprecated alias — use log_every


class TorchDistillTrainer:
    """
    Single-GPU distillation trainer.

    Works with any model whose ``forward(input_ids=...)`` returns an object
    with a ``.logits`` attribute — any HuggingFace CausalLM or compatible model.

    Accepts any iterable dataset: ``list[np.ndarray]``, a ``DataPipeline``,
    or a HuggingFace ``IterableDataset``.

    Args:
        student:   ``nn.Module`` whose forward returns ``.logits``.
        teachers:  ``TeacherRegistry`` — each teacher must already be loaded.
        config:    ``TorchTrainConfig``.
        alignment: Tokenizer alignment. Defaults to ``IdentityAlignment``.

    Example::

        from foundry.training import TorchDistillTrainer, TorchTrainConfig
        from foundry.teachers import TeacherRegistry
        from foundry.data import DataPipeline
        import numpy as np

        pipe    = DataPipeline(my_hf_dataset, tokenizer=tok, batch_size=8)
        trainer = TorchDistillTrainer(
            student  = my_model,
            teachers = TeacherRegistry.from_names(["org/teacher"]),
            config   = TorchTrainConfig(
                epochs=3, lr_scheduler="cosine", warmup_steps=100,
                torch_dtype="bfloat16", grad_accumulation_steps=4,
                save_every=500, save_dir="/ckpts/run1",
            ),
        )
        result = trainer.train(pipe, eval_dataset=my_eval_pipe)
        print(result["eval_losses"])
    """

    def __init__(
        self,
        student,
        teachers,
        config:    TorchTrainConfig | None = None,
        alignment = None,
    ) -> None:
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is required for TorchDistillTrainer. "
                "Install with: pip install olaverse-foundry[torch]"
            )
        self.student    = student
        self.teachers   = teachers
        self.cfg        = config or TorchTrainConfig()
        self.device     = self._resolve_device()
        self._dtype     = self._resolve_dtype()
        self.student.to(self.device)
        self._optimizer = self._build_optimizer()
        self._alignment = alignment

    # ── Setup ───────────────────────────────────────────────────────────────

    def _resolve_device(self):
        import torch
        d = self.cfg.device
        if d != "auto":
            return torch.device(d)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_dtype(self):
        import torch
        return {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }.get(self.cfg.torch_dtype, torch.float32)

    def _autocast(self):
        import torch
        if self._dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self._dtype)

    def _build_optimizer(self):
        import torch
        return torch.optim.AdamW(
            self.student.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    def _seed_everything(self) -> None:
        import torch
        seed = self.cfg.seed
        if not seed:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ── Checkpoint ──────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save student weights + optimizer state to ``<path>/checkpoint.pt``."""
        import torch
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        ckpt = p / "checkpoint.pt"
        torch.save({
            "model_state":     self.student.state_dict(),
            "optimizer_state": self._optimizer.state_dict(),
            "config":          vars(self.cfg),
        }, ckpt)
        return ckpt

    def resume_from_checkpoint(self, path: str | Path) -> None:
        """Load student weights + optimizer state from a checkpoint."""
        import torch
        p = Path(path)
        ckpt = p if p.suffix == ".pt" else p / "checkpoint.pt"
        data = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.student.load_state_dict(data["model_state"])
        self._optimizer.load_state_dict(data["optimizer_state"])

    # ── Eval ────────────────────────────────────────────────────────────────

    def _run_eval(self, eval_dataset) -> float:
        """Run CE loss over eval_dataset without updating gradients."""
        import torch
        import torch.nn.functional as F
        self.student.eval()
        total, n = 0.0, 0
        with torch.no_grad(), self._autocast():
            for batch_ids in eval_dataset:
                if isinstance(batch_ids, dict):
                    batch_ids = batch_ids["input_ids"]
                if not isinstance(batch_ids, np.ndarray):
                    batch_ids = np.array(batch_ids)
                if batch_ids.ndim == 1:
                    batch_ids = batch_ids[None, :]
                ids_t  = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
                out    = self.student(input_ids=ids_t)
                logits = out.logits.float()
                V      = logits.shape[-1]
                gold_t = torch.roll(ids_t, -1, dims=1)
                gold_t[:, -1] = 0
                loss = F.cross_entropy(
                    logits[:, :-1].contiguous().view(-1, V),
                    gold_t[:, :-1].contiguous().view(-1),
                    ignore_index=0,
                )
                total += float(loss.item())
                n     += 1
        self.student.train()
        return total / max(1, n)

    # ── Core training step ───────────────────────────────────────────────────

    def train_step(
        self,
        input_ids:      np.ndarray,
        *,
        is_first_accum: bool = True,
        is_last_accum:  bool = True,
    ) -> float:
        """
        One forward + backward + (conditional) optimizer step.

        Returns the unscaled combined loss as a Python float.
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from foundry.fusion.align import IdentityAlignment
        from foundry.fusion.strategies import STRATEGY_REGISTRY

        ids_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        n_acc = max(1, self.cfg.grad_accumulation_steps)

        if is_first_accum:
            self._optimizer.zero_grad()

        try:
            with self._autocast():
                out            = self.student(input_ids=ids_t)
                student_logits = out.logits.float()
                V              = student_logits.shape[-1]

                gold_t = torch.roll(ids_t, -1, dims=1)
                gold_t[:, -1] = 0

                ce_loss = F.cross_entropy(
                    student_logits[:, :-1].contiguous().view(-1, V),
                    gold_t[:, :-1].contiguous().view(-1),
                    ignore_index=0,
                )

                kl_loss = torch.tensor(0.0, device=self.device)

                if len(self.teachers) > 0:
                    align = self._alignment or IdentityAlignment()
                    fuse  = STRATEGY_REGISTRY.get(
                        self.cfg.fusion_strategy, STRATEGY_REGISTRY["min_ce"]
                    )
                    teacher_dists, teacher_weights = [], []
                    for teacher in self.teachers:
                        t_idx, t_prob = teacher.distribution(input_ids, top_k=self.cfg.top_k)
                        teacher_dists.append(align.map(t_idx, t_prob, V))
                        teacher_weights.append(teacher.weight)

                    gold_np  = np.roll(input_ids, -1, axis=1)
                    gold_np[:, -1] = 0
                    fused_np = fuse(teacher_dists, gold_np, teacher_weights)
                    fused_t  = torch.tensor(fused_np, dtype=torch.float32, device=self.device)
                    fused_t  = (fused_t + 1e-9) / (fused_t + 1e-9).sum(dim=-1, keepdim=True)

                    log_stu = F.log_softmax(student_logits[:, :-1], dim=-1)
                    kl_loss = F.kl_div(log_stu, fused_t[:, :-1],
                                       reduction="batchmean", log_target=False)

                raw_loss = self.cfg.alpha * ce_loss + (1.0 - self.cfg.alpha) * kl_loss
                (raw_loss / n_acc).backward()

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                import torch as _t
                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
                self._optimizer.zero_grad()
                raise RuntimeError(
                    "CUDA out of memory. Suggestions:\n"
                    "  • Reduce batch size\n"
                    f"  • Increase grad_accumulation_steps (currently {n_acc})\n"
                    "  • Set torch_dtype='bfloat16' in TorchTrainConfig\n"
                    "  • Enable gradient_checkpointing on the student\n"
                    f"Original: {exc}"
                ) from exc
            raise

        if is_last_accum:
            nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.max_grad_norm)
            self._optimizer.step()

        return float(raw_loss.item())

    # ── Training loop ────────────────────────────────────────────────────────

    def train(
        self,
        dataset,
        eval_dataset        = None,
        on_step:  Optional[Callable[[int, float], None]] = None,
        shuffle:  bool = False,
        total_steps: Optional[int] = None,
    ) -> dict:
        """
        Run the full training loop.

        Args:
            dataset:     Iterable of (batch, seq_len) int arrays, or a
                         ``DataPipeline`` / HF ``IterableDataset``.
            eval_dataset: Optional eval iterable. Evaluated every
                          ``config.eval_every`` optimizer steps.
            on_step:     Callback(global_step, loss) fired after each optimizer
                         step (respects log_every).
            shuffle:     Shuffle dataset at the start of each epoch (only works
                         for finite list-like datasets; streaming uses
                         ``DataPipeline.shuffle_buffer``).
            total_steps: Override the total optimizer step count used by the
                         LR scheduler. Required for streaming datasets.

        Returns:
            dict with "losses", "eval_losses" (dict step→loss), and "device".
        """
        self._seed_everything()

        # Compute total_steps for scheduler
        if total_steps is None:
            try:
                n_batches   = len(dataset)  # type: ignore[arg-type]
                total_steps = math.ceil(
                    n_batches * self.cfg.epochs
                    / max(1, self.cfg.grad_accumulation_steps)
                )
            except TypeError:
                total_steps = 0   # streaming — scheduler falls back to constant

        scheduler = build_scheduler(
            self._optimizer,
            self.cfg.lr_scheduler,
            self.cfg.warmup_steps,
            total_steps,
        )

        logger = _FoundryLogger(
            backend=self.cfg.log_backend,
            project=self.cfg.project,
            run_name=self.cfg.run_name,
            config=vars(self.cfg),
        )

        losses:      list[float]       = []
        eval_losses: dict[int, float]  = {}
        global_step: int               = 0
        accum_idx:   int               = 0
        n_acc        = max(1, self.cfg.grad_accumulation_steps)

        self.student.train()
        self._optimizer.zero_grad()

        try:
            for epoch in range(self.cfg.epochs):
                epoch_data = dataset
                if shuffle:
                    try:
                        epoch_data = list(dataset)
                        random.shuffle(epoch_data)
                    except TypeError:
                        pass  # streaming — can't shuffle in-place

                for batch_ids in epoch_data:
                    if isinstance(batch_ids, dict):
                        batch_ids = batch_ids["input_ids"]
                    if not isinstance(batch_ids, np.ndarray):
                        batch_ids = np.array(batch_ids)
                    if batch_ids.ndim == 1:
                        batch_ids = batch_ids[None, :]

                    pos      = accum_idx % n_acc
                    is_first = pos == 0
                    is_last  = pos == n_acc - 1

                    loss = self.train_step(
                        batch_ids,
                        is_first_accum=is_first,
                        is_last_accum=is_last,
                    )
                    losses.append(loss)
                    accum_idx += 1

                    if is_last:
                        if scheduler is not None:
                            scheduler.step()
                        logger.log(global_step, loss)
                        if on_step and global_step % self.cfg.log_every == 0:
                            on_step(global_step, loss)

                        # Eval
                        if (eval_dataset is not None
                                and self.cfg.eval_every > 0
                                and global_step % self.cfg.eval_every == 0):
                            ev = self._run_eval(eval_dataset)
                            eval_losses[global_step] = ev
                            logger.log_eval(global_step, ev)

                        # Auto-checkpoint
                        if (self.cfg.save_every > 0
                                and self.cfg.save_dir
                                and global_step > 0
                                and global_step % self.cfg.save_every == 0):
                            self.save_checkpoint(self.cfg.save_dir)

                        global_step += 1

            # Flush any incomplete accumulation window
            if accum_idx % n_acc != 0:
                import torch.nn as nn
                nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.max_grad_norm)
                self._optimizer.step()
                self._optimizer.zero_grad()

        finally:
            logger.finish()

        return {
            "losses":      losses,
            "eval_losses": eval_losses,
            "device":      str(self.device),
        }
