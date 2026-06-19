"""
DistilMLMTrainer — combined distillation + MLM (the DistilBERT objective).

Trains a student encoder against a teacher with a *single* multi-part loss, so the
distillation and the data-learning signals never fight each other (the failure mode
of running distillation then MLM sequentially):

    L = w_mlm · CE(student_mlm, masked_labels)              # learn from data
      + w_ce  · T² · KL(student_logits/T ‖ teacher_logits/T) # copy the teacher's soft predictions
      + w_cos · (1 − cos(student_hidden, teacher_hidden))    # align representations

This is exactly how DistilBERT was trained. It requires the student and teacher to
**share a vocabulary** (so the logit-level KL aligns) — which holds when the student
is warm-started from the teacher. Hidden sizes may differ (a projection is added).

Both student and teacher must be masked-LM models: ``forward(input_ids=...,
attention_mask=..., output_hidden_states=True)`` returns ``.logits`` (B,S,V) and
``.hidden_states`` (tuple; the last is the final encoder state). Any HF
``AutoModelForMaskedLM`` satisfies this.
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
class DistilMLMConfig:
    """Configuration for combined distillation + MLM (DistilBERT-style)."""

    # loss weights (DistilBERT defaults) + temperature
    mlm_weight:    float = 2.0
    distill_weight: float = 5.0
    cosine_weight: float = 1.0
    temperature:   float = 2.0

    # masking
    mask_prob:     float = 0.15
    mask_token_id: Optional[int] = None
    pad_token_id:  int   = 0
    vocab_size:    Optional[int] = None

    # standard training knobs
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


class DistilMLMTrainer:
    """Combined distillation + MLM trainer. See module docstring for the contract."""

    def __init__(self, student, teacher, tokenizer=None, config: DistilMLMConfig | None = None) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "torch is required for DistilMLMTrainer. "
                "Install with: pip install olaverse-foundry[torch]"
            )
        self.student   = student
        self.teacher   = teacher
        self.tokenizer = tokenizer
        self.cfg       = config or DistilMLMConfig()

        # masking vocabulary (mirrors MLMTrainer)
        mask_id, pad_id, vocab, specials = self.cfg.mask_token_id, self.cfg.pad_token_id, self.cfg.vocab_size, []
        if tokenizer is not None:
            mask_id = mask_id if mask_id is not None else getattr(tokenizer, "mask_token_id", None)
            tpad    = getattr(tokenizer, "pad_token_id", None)
            pad_id  = tpad if tpad is not None else pad_id
            vocab   = vocab if vocab is not None else (len(tokenizer) if hasattr(tokenizer, "__len__") else None)
            specials = list(getattr(tokenizer, "all_special_ids", []) or [])
        if mask_id is None:
            raise ValueError("DistilMLMTrainer needs a mask token id — pass a tokenizer or set "
                             "DistilMLMConfig.mask_token_id.")
        self._mask_id  = int(mask_id)
        self._pad_id   = int(pad_id)
        self._vocab    = int(vocab) if vocab is not None else None
        self._specials = set(int(s) for s in specials)

        self.device = self._resolve_device()
        self._dtype = self._resolve_dtype()
        self.student.to(self.device)
        self.teacher.to(self.device)
        self.teacher.eval()

        # hidden-state projection for the cosine loss when sizes differ
        self._projector: Any = None
        sd = getattr(getattr(student, "config", None), "hidden_size", None)
        td = getattr(getattr(teacher, "config", None), "hidden_size", None)
        if sd is not None and td is not None and sd != td:
            import torch.nn as nn
            self._projector = nn.Linear(int(sd), int(td), bias=False).to(self.device)
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
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}.get(self.cfg.torch_dtype, torch.float32)

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
        return torch.optim.AdamW(params, lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)

    def _seed_everything(self) -> None:
        import torch
        seed = self.cfg.seed
        if not seed:
            return
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ── Checkpoint ──────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save student (+ projector) + optimizer state to ``<path>/checkpoint.pt``."""
        import torch
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        ckpt = p / "checkpoint.pt"
        payload: dict = {"model_state": self.student.state_dict(),
                         "optimizer_state": self._optimizer.state_dict(),
                         "config": vars(self.cfg)}
        if self._projector is not None:
            payload["projector_state"] = self._projector.state_dict()
        torch.save(payload, ckpt)
        return ckpt

    def resume_from_checkpoint(self, path: str | Path) -> None:
        """Load student (+ projector) + optimizer state from a checkpoint."""
        import torch
        p = Path(path)
        ckpt = p if p.suffix == ".pt" else p / "checkpoint.pt"
        data = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.student.load_state_dict(data["model_state"])
        self._optimizer.load_state_dict(data["optimizer_state"])
        if self._projector is not None and "projector_state" in data:
            self._projector.load_state_dict(data["projector_state"])

    # ── Masking ───────────────────────────────────────────────────────────────

    def _to_ids_mask(self, batch):
        import torch
        if isinstance(batch, dict):
            ids_np  = np.asarray(batch["input_ids"])
            mask_np = batch.get("attention_mask")
            mask_np = np.asarray(mask_np) if mask_np is not None else (ids_np != self._pad_id)
        else:
            ids_np  = np.asarray(batch)
            mask_np = (ids_np != self._pad_id)
        if ids_np.ndim == 1:
            ids_np, mask_np = ids_np[None, :], mask_np[None, :]
        return (torch.tensor(ids_np,  dtype=torch.long, device=self.device),
                torch.tensor(mask_np, dtype=torch.long, device=self.device))

    def _mask_tokens(self, ids_t, attn_t):
        import torch
        labels = ids_t.clone()
        prob   = torch.full(labels.shape, self.cfg.mask_prob, device=self.device)
        special = torch.zeros_like(ids_t, dtype=torch.bool)
        for sid in self._specials:
            special |= (ids_t == sid)
        special |= (attn_t == 0)
        prob.masked_fill_(special, 0.0)
        masked = torch.bernoulli(prob).bool()
        if not bool(masked.any()):                    # guarantee ≥1 masked → no NaN CE
            valid = (~special).view(-1).nonzero(as_tuple=False).flatten()
            if valid.numel() > 0:
                masked.view(-1)[valid[torch.randint(valid.numel(), (1,), device=self.device)]] = True
        labels[~masked] = -100

        inputs = ids_t.clone()
        repl = torch.bernoulli(torch.full(labels.shape, 0.8, device=self.device)).bool() & masked
        inputs[repl] = self._mask_id
        if self._vocab:
            rand = torch.bernoulli(torch.full(labels.shape, 0.5, device=self.device)).bool() & masked & ~repl
            inputs[rand] = torch.randint(self._vocab, labels.shape, dtype=torch.long, device=self.device)[rand]
        return inputs, labels

    # ── Combined loss ─────────────────────────────────────────────────────────

    def _compute_loss(self, inputs, attn_t, labels):
        import torch
        import torch.nn.functional as F

        s_out    = self.student(input_ids=inputs, attention_mask=attn_t, output_hidden_states=True)
        s_logits = s_out.logits.float()
        s_hidden = s_out.hidden_states[-1].float()
        with torch.no_grad():
            t_out    = self.teacher(input_ids=inputs, attention_mask=attn_t, output_hidden_states=True)
            t_logits = t_out.logits.float()
            t_hidden = t_out.hidden_states[-1].float()

        if s_logits.shape[-1] != t_logits.shape[-1]:
            raise ValueError(
                f"Student vocab ({s_logits.shape[-1]}) != teacher vocab ({t_logits.shape[-1]}). "
                "Logit-level distillation needs a shared vocabulary (warm-start the student "
                "from the teacher)."
            )
        if self._projector is not None:
            s_hidden = self._projector(s_hidden)

        V      = s_logits.shape[-1]
        masked = labels != -100

        # 1. MLM cross-entropy on masked tokens
        l_mlm = F.cross_entropy(s_logits.view(-1, V), labels.view(-1), ignore_index=-100)

        # 2. Soft-label KL distillation on masked tokens (temperature-scaled)
        T = self.cfg.temperature
        s_m = s_logits[masked]
        t_m = t_logits[masked]
        l_ce = F.kl_div(F.log_softmax(s_m / T, dim=-1),
                        F.softmax(t_m / T, dim=-1),
                        reduction="batchmean") * (T * T)

        # 3. Cosine alignment of final hidden states over real (non-pad) tokens
        tok = attn_t.bool().view(-1)
        sh  = s_hidden.reshape(-1, s_hidden.shape[-1])[tok]
        th  = t_hidden.reshape(-1, t_hidden.shape[-1])[tok]
        l_cos = (1.0 - F.cosine_similarity(sh, th, dim=-1)).mean()

        loss = (self.cfg.mlm_weight * l_mlm
                + self.cfg.distill_weight * l_ce
                + self.cfg.cosine_weight * l_cos)
        return loss

    # ── Step ────────────────────────────────────────────────────────────────────

    def train_step(self, batch, *, is_first_accum=True, is_last_accum=True) -> float:
        import torch
        import torch.nn as nn
        n_acc = max(1, self.cfg.grad_accumulation_steps)
        if is_first_accum:
            self._optimizer.zero_grad()
        ids_t, attn_t  = self._to_ids_mask(batch)
        inputs, labels = self._mask_tokens(ids_t, attn_t)
        try:
            with self._autocast():
                loss = self._compute_loss(inputs, attn_t, labels)
            if not torch.isfinite(loss):
                self._optimizer.zero_grad()
                return float(loss.item())
            (loss / n_acc).backward()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._optimizer.zero_grad()
                raise RuntimeError(
                    "CUDA out of memory. Reduce batch size / max_length, raise "
                    f"grad_accumulation_steps (now {n_acc}), or use bfloat16.\nOriginal: {exc}"
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
                ids_t, attn_t  = self._to_ids_mask(batch)
                inputs, labels = self._mask_tokens(ids_t, attn_t)
                total += float(self._compute_loss(inputs, attn_t, labels).item()); n += 1
        self.student.train()
        return total / max(1, n)

    # ── Training loop ────────────────────────────────────────────────────────────

    def train(self, dataset, eval_dataset=None,
              on_step: Optional[Callable[[int, float], None]] = None,
              shuffle: bool = False, total_steps: Optional[int] = None) -> dict:
        """Run combined distill+MLM for ``config.epochs`` epochs. Returns
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
        scheduler = build_scheduler(self._optimizer, self.cfg.lr_scheduler,
                                    self.cfg.warmup_steps, total_steps)
        logger = _FoundryLogger(backend=self.cfg.log_backend, project=self.cfg.project,
                                run_name=self.cfg.run_name, config=vars(self.cfg))

        losses: list[float] = []
        eval_losses: dict[int, float] = {}
        global_step = accum_idx = 0
        self.student.train(); self._optimizer.zero_grad()
        try:
            for _epoch in range(self.cfg.epochs):
                idxs = list(range(len(data_list)))
                if shuffle:
                    random.shuffle(idxs)
                for i in idxs:
                    pos = accum_idx % n_acc
                    loss = self.train_step(data_list[i],
                                           is_first_accum=(pos == 0),
                                           is_last_accum=(pos == n_acc - 1))
                    losses.append(loss); accum_idx += 1
                    if pos == n_acc - 1:
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
                self._optimizer.step(); self._optimizer.zero_grad()
        finally:
            logger.finish()
        return {"losses": losses, "eval_losses": eval_losses, "device": str(self.device)}
