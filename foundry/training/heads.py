"""
Head trainers — fine-tune task heads on top of a base encoder.

Once you have a base encoder (from MLMTrainer or EncoderDistillTrainer), you bolt
heads onto it for downstream tasks:

  • SequenceClassificationTrainer — classification, language-ID, moderation,
    sentiment, and reranking-as-classification. (B,) labels.
  • TokenClassificationTrainer    — NER and other token-level tasks. (B,S) labels,
    pad/subword positions = -100.

Both are **model-agnostic**: they accept any model whose
``forward(input_ids=..., attention_mask=...)`` returns an object with a
``.logits`` tensor — any HF ``AutoModelForSequenceClassification`` /
``AutoModelForTokenClassification``, or your own custom head module.

Frozen-backbone / shared-encoder path
--------------------------------------
Set ``freeze_backbone=True`` to train only the head (a cheap per-task adapter)
while the encoder stays frozen. Several heads can then share one base encoder /
one forward pass — the on-device serving story. Use :func:`build_encoder_with_head`
to attach a fresh head to a saved base in one line.

Shared production features: mixed precision, grad accumulation, LR schedule +
warmup, eval loop (loss + accuracy), checkpoint save/resume, auto-checkpoint,
OOM message, logging. Consume ``DataPipeline(..., label_column=...)`` batches.
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def freeze_backbone(model, head_keywords=("classifier", "score", "head")):
    """
    Freeze every parameter except the task head (params whose name contains one
    of ``head_keywords``). Returns ``(model, n_trainable, n_frozen)``.
    """
    n_train = n_frozen = 0
    for name, p in model.named_parameters():
        is_head = any(k in name.lower() for k in head_keywords)
        p.requires_grad_(is_head)
        if is_head:
            n_train += p.numel()
        else:
            n_frozen += p.numel()
    return model, n_train, n_frozen


def build_encoder_with_head(base, num_labels: int, task: str = "sequence", **kwargs):
    """
    Attach a fresh classification head to a saved base encoder (or model id).

    Args:
        base:       Path/id of a base encoder (e.g. './afri-base-mlm').
        num_labels: Number of classes / tags.
        task:       "sequence" (AutoModelForSequenceClassification) or
                    "token" (AutoModelForTokenClassification).

    Returns the HF model with a randomly-initialised head over the base weights.
    """
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
        )
    except ImportError:
        raise ImportError(
            "transformers is required for build_encoder_with_head. "
            "Install with: pip install olaverse-foundry[torch]"
        )
    cls = AutoModelForSequenceClassification if task == "sequence" else AutoModelForTokenClassification
    return cls.from_pretrained(base, num_labels=num_labels, **kwargs)


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class HeadTrainConfig:
    """Configuration shared by the sequence- and token-classification trainers."""

    num_labels:              int   = 2
    multi_label:             bool  = False   # sequence task only → BCEWithLogits
    freeze_backbone:         bool  = False
    pad_token_id:            int   = 0

    learning_rate:           float = 2e-5
    epochs:                  int   = 3
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


# ── Shared base ───────────────────────────────────────────────────────────────

class _HeadTrainer:
    """Common machinery for head fine-tuning. Subclasses define the loss + metric."""

    def __init__(self, model, config: HeadTrainConfig | None = None) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "torch is required for head trainers. "
                "Install with: pip install olaverse-foundry[torch]"
            )
        self.model = model
        self.cfg   = config or HeadTrainConfig()
        if self.cfg.freeze_backbone:
            freeze_backbone(self.model)
        self.device = self._resolve_device()
        self._dtype = self._resolve_dtype()
        self.model.to(self.device)
        self._optimizer = self._build_optimizer()

    # setup
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
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.cfg.learning_rate,
                                 weight_decay=self.cfg.weight_decay)

    def _seed_everything(self) -> None:
        import torch
        seed = self.cfg.seed
        if not seed:
            return
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # checkpoint
    def save_checkpoint(self, path: str | Path) -> Path:
        """Save model + optimizer state to ``<path>/checkpoint.pt``."""
        import torch
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        ckpt = p / "checkpoint.pt"
        torch.save({"model_state": self.model.state_dict(),
                    "optimizer_state": self._optimizer.state_dict(),
                    "config": vars(self.cfg)}, ckpt)
        return ckpt

    def resume_from_checkpoint(self, path: str | Path) -> None:
        """Load model + optimizer state from a checkpoint."""
        import torch
        p = Path(path)
        ckpt = p if p.suffix == ".pt" else p / "checkpoint.pt"
        data = torch.load(ckpt, map_location=self.device, weights_only=True)
        self.model.load_state_dict(data["model_state"])
        self._optimizer.load_state_dict(data["optimizer_state"])

    # batch
    def _to_batch(self, batch):
        import torch
        if not isinstance(batch, dict) or "labels" not in batch:
            raise ValueError(
                "Head trainers need batches with 'input_ids', 'attention_mask', "
                "and 'labels'. Use DataPipeline(..., label_column=...)."
            )
        ids  = np.asarray(batch["input_ids"])
        mask = batch.get("attention_mask")
        mask = np.asarray(mask) if mask is not None else (ids != self.cfg.pad_token_id)
        lab  = np.asarray(batch["labels"])
        ids_t  = torch.tensor(ids,  dtype=torch.long, device=self.device)
        mask_t = torch.tensor(mask, dtype=torch.long, device=self.device)
        lab_t  = torch.tensor(lab,  dtype=torch.long, device=self.device)
        return ids_t, mask_t, lab_t

    # subclass hooks
    def _loss(self, logits, labels, attn):
        raise NotImplementedError

    def _correct(self, logits, labels):
        """Return (n_correct, n_total) for accuracy."""
        raise NotImplementedError

    # step
    def train_step(self, batch, *, is_first_accum=True, is_last_accum=True) -> float:
        import torch
        import torch.nn as nn
        n_acc = max(1, self.cfg.grad_accumulation_steps)
        if is_first_accum:
            self._optimizer.zero_grad()
        ids_t, mask_t, lab_t = self._to_batch(batch)
        try:
            with self._autocast():
                logits = self.model(input_ids=ids_t, attention_mask=mask_t).logits.float()
                loss   = self._loss(logits, lab_t, mask_t)
                (loss / n_acc).backward()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._optimizer.zero_grad()
                raise RuntimeError(
                    "CUDA out of memory. Reduce batch size / max_length, raise "
                    f"grad_accumulation_steps (now {n_acc}), or use bfloat16.\n"
                    f"Original: {exc}"
                ) from exc
            raise
        if is_last_accum:
            params = [p for p in self.model.parameters() if p.requires_grad]
            nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
            self._optimizer.step()
        return float(loss.item())

    def _run_eval(self, eval_dataset):
        import torch
        self.model.eval()
        total_loss, n_batches, correct, total = 0.0, 0, 0, 0
        with torch.no_grad(), self._autocast():
            for batch in eval_dataset:
                ids_t, mask_t, lab_t = self._to_batch(batch)
                logits = self.model(input_ids=ids_t, attention_mask=mask_t).logits.float()
                total_loss += float(self._loss(logits, lab_t, mask_t).item()); n_batches += 1
                c, t = self._correct(logits, lab_t)
                correct += c; total += t
        self.model.train()
        return total_loss / max(1, n_batches), correct / max(1, total)

    def predict(self, dataset):
        """Run the model over ``dataset`` and return ``(preds, labels)`` as flat
        numpy arrays (token tasks drop -100 positions). Used by the eval harness
        to compute accuracy / macro-F1."""
        import torch
        self.model.eval()
        all_p, all_l = [], []
        with torch.no_grad(), self._autocast():
            for batch in dataset:
                ids_t, mask_t, lab_t = self._to_batch(batch)
                logits = self.model(input_ids=ids_t, attention_mask=mask_t).logits.float()
                all_p.append(logits.argmax(dim=-1).cpu().numpy().reshape(-1))
                all_l.append(lab_t.cpu().numpy().reshape(-1))
        self.model.train()
        preds  = np.concatenate(all_p)
        labels = np.concatenate(all_l)
        keep   = labels != -100
        return preds[keep], labels[keep]

    # loop
    def train(self, dataset, eval_dataset=None,
              on_step: Optional[Callable[[int, float], None]] = None,
              shuffle: bool = False, total_steps: Optional[int] = None) -> dict:
        """Fine-tune for ``config.epochs``. Returns
        ``{"losses", "eval_losses", "eval_metrics", "device"}`` (eval_metrics = accuracy)."""
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
        eval_losses:  dict[int, float] = {}
        eval_metrics: dict[int, float] = {}
        global_step = accum_idx = 0
        self.model.train(); self._optimizer.zero_grad()
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
                            ev, acc = self._run_eval(eval_dataset)
                            eval_losses[global_step] = ev
                            eval_metrics[global_step] = acc
                            logger.log_eval(global_step, ev)
                        if (self.cfg.save_every > 0 and self.cfg.save_dir
                                and global_step > 0 and global_step % self.cfg.save_every == 0):
                            self.save_checkpoint(self.cfg.save_dir)
                        global_step += 1
            if accum_idx % n_acc != 0:
                import torch.nn as nn
                params = [p for p in self.model.parameters() if p.requires_grad]
                nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
                self._optimizer.step(); self._optimizer.zero_grad()
        finally:
            logger.finish()
        return {"losses": losses, "eval_losses": eval_losses,
                "eval_metrics": eval_metrics, "device": str(self.device)}


# ── Sequence classification (classifier / langID / moderation / reranker) ──────

class SequenceClassificationTrainer(_HeadTrainer):
    """Fine-tune a sequence-classification head. Labels are ``(B,)`` class ids
    (or ``(B, num_labels)`` floats when ``multi_label=True``)."""

    def _loss(self, logits, labels, attn):
        import torch.nn.functional as F
        if self.cfg.multi_label:
            return F.binary_cross_entropy_with_logits(logits, labels.float())
        return F.cross_entropy(logits, labels)

    def _correct(self, logits, labels):
        if self.cfg.multi_label:
            pred = (logits > 0).long()
            return int((pred == labels).all(dim=-1).sum()), labels.shape[0]
        pred = logits.argmax(dim=-1)
        return int((pred == labels).sum()), labels.shape[0]


# ── Token classification (NER) ─────────────────────────────────────────────────

class TokenClassificationTrainer(_HeadTrainer):
    """Fine-tune a token-classification head. Labels are ``(B, S)`` tag ids with
    ``-100`` at pad/subword positions (ignored in the loss)."""

    def _loss(self, logits, labels, attn):
        import torch.nn.functional as F
        C = logits.shape[-1]
        return F.cross_entropy(logits.view(-1, C), labels.view(-1), ignore_index=-100)

    def _correct(self, logits, labels):
        pred  = logits.argmax(dim=-1)
        valid = labels != -100
        return int(((pred == labels) & valid).sum()), int(valid.sum())
