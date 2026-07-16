"""
EncoderDistillTrainer — token-level (hidden-state) distillation for encoders.

The DistilBERT / MiniLM-style path: a small student encoder learns to reproduce
a strong teacher encoder's **per-token hidden states**, not just the pooled
sentence vector. Because every token position is supervised, the resulting base
keeps token-level representations — so heads that need them (NER, token
classification) work, unlike a pooled-only embedding distillation.

Use this to compress a strong existing encoder (e.g. AfroXLMR, mmBERT, BGE) into
your own smaller architecture when you have limited raw text — distillation is
far more data-efficient than MLM-from-scratch.

Contracts
---------
* student: ``student(input_ids=..., attention_mask=...)`` returns an object with
  ``.last_hidden_state`` of shape ``(B, S, D_student)``.
* teacher: same call shape, returns ``.last_hidden_state`` of shape
  ``(B, S, D_teacher)`` (any HF ``AutoModel`` encoder). If the dims differ, a
  trainable linear projection ``D_student → D_teacher`` is added automatically.

Production features mirror the other trainers: mixed precision, grad
accumulation, LR schedule + warmup, eval loop, checkpoint save/resume,
auto-checkpoint, OOM message, logging.
"""
from __future__ import annotations

import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from foundry.training._logger import _FoundryLogger
from foundry.training._scheduler import build_scheduler


@dataclass
class EncoderDistillConfig:
    """Configuration for token-level encoder distillation."""

    loss:                    str   = "mse"   # "mse" | "cosine" on per-token hidden states
    pad_token_id:            int   = 0

    learning_rate:           float = 5e-5
    epochs:                  int   = 1
    weight_decay:            float = 0.01
    max_grad_norm:           float = 1.0
    device:                  str   = "auto"
    grad_accumulation_steps: int   = 1
    torch_dtype:             str   = "float32"
    lr_scheduler:            str   = "cosine"
    warmup_steps:            int   = 0
    eval_every:              int   = 0
    save_every:              int   = 0
    save_dir:                str   = ""
    log_every:               int   = 50
    log_backend:             str   = "none"
    run_name:                str   = ""
    project:                 str   = "olaverse-foundry"
    seed:                    int   = 42


class EncoderDistillTrainer:
    """Token-level hidden-state distillation. See module docstring for contracts."""

    def __init__(self, student, teacher, config: EncoderDistillConfig | None = None) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "torch is required for EncoderDistillTrainer. "
                "Install with: pip install olaverse-foundry[torch]"
            )
        self.student = student
        self.teacher = teacher
        self.cfg     = config or EncoderDistillConfig()
        self.device  = self._resolve_device()
        self._dtype  = self._resolve_dtype()
        self.student.to(self.device)
        if hasattr(teacher, "to"):
            teacher.to(self.device)
        if hasattr(teacher, "eval"):
            teacher.eval()
        self._projector: Any = None
        # Eagerly add the student→teacher projection when both hidden sizes are
        # known (the usual case for HF encoders) so the optimizer's param groups
        # are final before the LR scheduler is built. Otherwise fall back to lazy
        # creation on the first forward and defer the scheduler build.
        sd = getattr(getattr(student, "config", None), "hidden_size", None)
        td = getattr(getattr(teacher, "config", None), "hidden_size", None)
        if sd is not None and td is not None:
            if sd != td:
                import torch.nn as nn
                self._projector = nn.Linear(int(sd), int(td), bias=False).to(self.device)
            self._defer_scheduler = False
        else:
            self._defer_scheduler = True
        self._optimizer = self._build_optimizer()

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
        params = list(self.student.parameters())
        if self._projector is not None:
            params += list(self._projector.parameters())
        return torch.optim.AdamW(
            params,
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

    def _ensure_projector(self, d_student: int, d_teacher: int) -> None:
        """Lazily add a student→teacher projection when hidden sizes differ."""
        if d_student == d_teacher or self._projector is not None:
            return
        import torch.nn as nn
        self._projector = nn.Linear(d_student, d_teacher, bias=False).to(self.device)
        self._optimizer.add_param_group({"params": self._projector.parameters()})

    # ── Checkpoint ──────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save student weights (+ projector) + optimizer state to ``<path>/checkpoint.pt``."""
        import torch
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        ckpt = p / "checkpoint.pt"
        payload: dict = {
            "model_state":     self.student.state_dict(),
            "optimizer_state": self._optimizer.state_dict(),
            "config":          vars(self.cfg),
        }
        if self._projector is not None:
            payload["projector_state"] = self._projector.state_dict()
        torch.save(payload, ckpt)
        return ckpt

    def resume_from_checkpoint(self, path: str | Path) -> None:
        """Load student weights (+ projector) + optimizer state from a checkpoint."""
        import torch
        p = Path(path)
        ckpt = p if p.suffix == ".pt" else p / "checkpoint.pt"
        data = torch.load(ckpt, map_location=self.device, weights_only=True)
        self.student.load_state_dict(data["model_state"])
        self._optimizer.load_state_dict(data["optimizer_state"])
        if self._projector is not None and "projector_state" in data:
            self._projector.load_state_dict(data["projector_state"])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_ids_mask(self, batch):
        import torch
        if isinstance(batch, dict):
            ids_np  = np.asarray(batch["input_ids"])
            mask_np = batch.get("attention_mask")
            mask_np = np.asarray(mask_np) if mask_np is not None else (ids_np != self.cfg.pad_token_id)
        else:
            ids_np  = np.asarray(batch)
            mask_np = (ids_np != self.cfg.pad_token_id)
        if ids_np.ndim == 1:
            ids_np, mask_np = ids_np[None, :], mask_np[None, :]
        ids_t  = torch.tensor(ids_np,  dtype=torch.long, device=self.device)
        mask_t = torch.tensor(mask_np, dtype=torch.long, device=self.device)
        return ids_t, mask_t

    @staticmethod
    def _hidden(out):
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out

    def _compute_loss(self, ids_t, attn_t):
        import torch
        import torch.nn.functional as F

        s_hidden = self._hidden(self.student(input_ids=ids_t, attention_mask=attn_t))
        with torch.no_grad():
            t_hidden = self._hidden(self.teacher(input_ids=ids_t, attention_mask=attn_t)).float()

        self._ensure_projector(s_hidden.shape[-1], t_hidden.shape[-1])
        if self._projector is not None:
            s_hidden = self._projector(s_hidden)
        s_hidden = s_hidden.float()

        mask = attn_t.unsqueeze(-1).float()               # (B, S, 1)
        denom = mask.sum().clamp(min=1.0)
        if self.cfg.loss == "cosine":
            cos = F.cosine_similarity(s_hidden, t_hidden, dim=-1)   # (B, S)
            tok_mask = attn_t.float()
            return ((1.0 - cos) * tok_mask).sum() / tok_mask.sum().clamp(min=1.0)
        # token-masked MSE, averaged over real (non-pad) elements
        se = (s_hidden - t_hidden) ** 2 * mask
        return se.sum() / (denom * s_hidden.shape[-1])

    # ── Core step ──────────────────────────────────────────────────────────────

    def train_step(self, batch, *, is_first_accum=True, is_last_accum=True) -> float:
        import torch
        import torch.nn as nn

        n_acc = max(1, self.cfg.grad_accumulation_steps)
        if is_first_accum:
            self._optimizer.zero_grad()

        ids_t, attn_t = self._to_ids_mask(batch)
        try:
            with self._autocast():
                loss = self._compute_loss(ids_t, attn_t)
                (loss / n_acc).backward()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._optimizer.zero_grad()
                raise RuntimeError(
                    "CUDA out of memory. Suggestions:\n"
                    "  • Reduce batch size or max_length\n"
                    f"  • Increase grad_accumulation_steps (currently {n_acc})\n"
                    "  • Set torch_dtype='bfloat16' in EncoderDistillConfig\n"
                    f"Original: {exc}"
                ) from exc
            raise

        if is_last_accum:
            params = list(self.student.parameters())
            if self._projector is not None:
                params += list(self._projector.parameters())
            nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
            self._optimizer.step()
        return float(loss.item())

    def _run_eval(self, eval_dataset) -> float:
        import torch
        self.student.eval()
        total, n = 0.0, 0
        with torch.no_grad(), self._autocast():
            for batch in eval_dataset:
                ids_t, attn_t = self._to_ids_mask(batch)
                total += float(self._compute_loss(ids_t, attn_t).item()); n += 1
        self.student.train()
        return total / max(1, n)

    # ── Training loop ────────────────────────────────────────────────────────────

    def train(
        self,
        dataset,
        eval_dataset = None,
        on_step:  Optional[Callable[[int, float], None]] = None,
        shuffle:  bool = False,
        total_steps: Optional[int] = None,
    ) -> dict:
        """Run distillation for ``config.epochs`` epochs. Returns
        ``{"losses", "eval_losses", "device"}``."""
        self._seed_everything()

        try:
            data_list = list(dataset)
        except Exception:
            data_list = dataset

        n_acc = max(1, self.cfg.grad_accumulation_steps)
        if total_steps is None:
            try:
                total_steps = math.ceil(len(data_list) * self.cfg.epochs / n_acc)
            except TypeError:
                total_steps = 0

        # If the projector was created eagerly (hidden sizes known), the param
        # groups are final → build the scheduler now (correct warmup). Otherwise
        # the projector may be added on the first forward, so defer the scheduler
        # build to the first optimizer step so it sees the final param groups.
        scheduler = None
        sched_ready = False
        if not self._defer_scheduler:
            scheduler = build_scheduler(
                self._optimizer, self.cfg.lr_scheduler, self.cfg.warmup_steps, total_steps
            )
            sched_ready = True
        logger = _FoundryLogger(
            backend=self.cfg.log_backend, project=self.cfg.project,
            run_name=self.cfg.run_name, config=vars(self.cfg),
        )

        losses: list[float]      = []
        eval_losses: dict[int, float] = {}
        global_step = 0
        accum_idx   = 0

        self.student.train()
        self._optimizer.zero_grad()
        try:
            for _epoch in range(self.cfg.epochs):
                indices = list(range(len(data_list)))
                if shuffle:
                    random.shuffle(indices)
                for i in indices:
                    pos      = accum_idx % n_acc
                    is_first = pos == 0
                    is_last  = pos == n_acc - 1
                    loss = self.train_step(
                        data_list[i], is_first_accum=is_first, is_last_accum=is_last
                    )
                    losses.append(loss)
                    accum_idx += 1
                    if is_last:
                        if not sched_ready:   # build once, after projector exists
                            scheduler = build_scheduler(
                                self._optimizer, self.cfg.lr_scheduler,
                                self.cfg.warmup_steps, total_steps,
                            )
                            sched_ready = True
                        if scheduler is not None:
                            scheduler.step()
                        logger.log(global_step, loss)
                        if on_step and global_step % self.cfg.log_every == 0:
                            on_step(global_step, loss)
                        if (eval_dataset is not None and self.cfg.eval_every > 0
                                and global_step % self.cfg.eval_every == 0):
                            ev = self._run_eval(eval_dataset)
                            eval_losses[global_step] = ev
                            logger.log_eval(global_step, ev)
                        if (self.cfg.save_every > 0 and self.cfg.save_dir
                                and global_step > 0 and global_step % self.cfg.save_every == 0):
                            self.save_checkpoint(self.cfg.save_dir)
                        global_step += 1

            if accum_idx % n_acc != 0:
                import torch.nn as nn
                params = list(self.student.parameters())
                if self._projector is not None:
                    params += list(self._projector.parameters())
                nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
                self._optimizer.step()
                self._optimizer.zero_grad()
        finally:
            logger.finish()

        return {"losses": losses, "eval_losses": eval_losses, "device": str(self.device)}
