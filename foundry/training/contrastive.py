"""
ContrastiveTrainer — InfoNCE / MultipleNegativesRanking for (cross-lingual) retrieval.

Turns an encoder into an embedding model for retrieval. Trains on **pairs**:
``{"anchor": text, "positive": text}`` (optionally ``"negative": text`` as a hard
negative). For each anchor, its positive is the correct match and every *other*
positive in the batch is an in-batch negative — the standard MultipleNegativesRanking
(InfoNCE) loss that powers e5 / bge-m3 / LaBSE.

For **cross-lingual** retrieval, make anchor and positive different languages
(parallel sentences) to align the multilingual space, and/or query↔passage pairs
for retrieval quality. Bigger ``batch_size`` = more in-batch negatives = better.

Model contract: ``model(input_ids=..., attention_mask=...)`` returns
``.last_hidden_state`` (any HF ``AutoModel`` encoder).
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


def _pool(hidden, attention_mask, mode: str):
    import torch  # noqa: F401
    if mode == "cls":
        return hidden[:, 0]
    m = attention_mask.unsqueeze(-1).float()
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive (InfoNCE) retrieval training."""

    pool:          str   = "mean"     # "mean" | "cls"
    temperature:   float = 0.05       # scales the similarity logits
    normalize:     bool  = True       # L2-normalise embeddings (cosine similarity)
    batch_size:    int   = 32         # in-batch negatives = batch_size − 1
    max_length:    int   = 128
    anchor_key:    str   = "anchor"
    positive_key:  str   = "positive"
    negative_key:  str   = "negative"  # optional hard negative

    learning_rate:           float = 2e-5
    epochs:                  int   = 1
    weight_decay:            float = 0.01
    max_grad_norm:           float = 1.0
    device:                  str   = "auto"
    grad_accumulation_steps: int   = 1
    torch_dtype:             str   = "float32"
    lr_scheduler:            str   = "cosine"
    warmup_steps:            int   = 0
    save_every:              int   = 0
    save_dir:                str   = ""
    log_every:               int   = 50
    log_backend:             str   = "none"
    run_name:                str   = ""
    project:                 str   = "olaverse-foundry"
    seed:                    int   = 42


class ContrastiveTrainer:
    """InfoNCE / MultipleNegativesRanking trainer. See module docstring for the contract."""

    def __init__(self, model, tokenizer, config: ContrastiveConfig | None = None) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "torch is required for ContrastiveTrainer. "
                "Install with: pip install olaverse-foundry[torch]"
            )
        if tokenizer is None:
            raise ValueError("ContrastiveTrainer needs a tokenizer to encode text pairs.")
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = config or ContrastiveConfig()
        self.device    = self._resolve_device()
        self._dtype    = self._resolve_dtype()
        self.model.to(self.device)
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
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate,
                                 weight_decay=self.cfg.weight_decay)

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
        data = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(data["model_state"])
        self._optimizer.load_state_dict(data["optimizer_state"])

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self, texts):
        """Tokenize + encode a list of strings → (N, D) embeddings (with grad)."""
        import torch
        enc = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=self.cfg.max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        emb = _pool(out.last_hidden_state, enc["attention_mask"], self.cfg.pool)
        if self.cfg.normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

    # ── Step ────────────────────────────────────────────────────────────────────

    def train_step(self, pairs, *, is_first_accum=True, is_last_accum=True) -> float:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        n_acc = max(1, self.cfg.grad_accumulation_steps)
        if is_first_accum:
            self._optimizer.zero_grad()

        a_key, p_key, n_key = self.cfg.anchor_key, self.cfg.positive_key, self.cfg.negative_key
        anchors   = [p[a_key] for p in pairs]
        positives = [p[p_key] for p in pairs]
        has_neg   = all(n_key in p and p[n_key] for p in pairs)

        try:
            with self._autocast():
                A = self.encode(anchors)              # (B, D)
                P = self.encode(positives)            # (B, D)
                cands = P
                if has_neg:
                    N = self.encode([p[n_key] for p in pairs])   # (B, D)
                    cands = torch.cat([P, N], dim=0)             # (2B, D)
                scores = (A @ cands.t()) / self.cfg.temperature  # (B, B or 2B)
                labels = torch.arange(len(pairs), device=self.device)
                loss   = F.cross_entropy(scores, labels)
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
                    "CUDA out of memory. Reduce batch_size / max_length, raise "
                    f"grad_accumulation_steps (now {n_acc}), or use bfloat16.\nOriginal: {exc}"
                ) from exc
            raise

        if is_last_accum:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self._optimizer.step()
        return float(loss.item())

    # ── Training loop ────────────────────────────────────────────────────────────

    def train(self, dataset, on_step: Optional[Callable[[int, float], None]] = None,
              shuffle: bool = True, total_steps: Optional[int] = None) -> dict:
        """Train on an iterable of pair-dicts for ``config.epochs`` epochs.
        Batches into groups of ``config.batch_size`` (the in-batch negative pool).
        Returns ``{"losses", "device"}``."""
        self._seed_everything()
        pairs = list(dataset)
        bs    = max(2, self.cfg.batch_size)
        n_acc = max(1, self.cfg.grad_accumulation_steps)

        n_batches = math.ceil(len(pairs) / bs)
        if total_steps is None:
            total_steps = math.ceil(n_batches * self.cfg.epochs / n_acc)
        scheduler = build_scheduler(self._optimizer, self.cfg.lr_scheduler,
                                    self.cfg.warmup_steps, total_steps)
        logger = _FoundryLogger(backend=self.cfg.log_backend, project=self.cfg.project,
                                run_name=self.cfg.run_name, config=vars(self.cfg))

        losses: list[float] = []
        global_step = accum_idx = 0
        self.model.train(); self._optimizer.zero_grad()
        try:
            for _epoch in range(self.cfg.epochs):
                order = list(range(len(pairs)))
                if shuffle:
                    random.shuffle(order)
                for b in range(0, len(order), bs):
                    batch = [pairs[i] for i in order[b:b + bs]]
                    if len(batch) < 2:           # need ≥2 for in-batch negatives
                        continue
                    pos = accum_idx % n_acc
                    loss = self.train_step(batch, is_first_accum=(pos == 0),
                                           is_last_accum=(pos == n_acc - 1))
                    losses.append(loss); accum_idx += 1
                    if pos == n_acc - 1:
                        if scheduler is not None:
                            scheduler.step()
                        logger.log(global_step, loss)
                        if on_step and global_step % self.cfg.log_every == 0:
                            on_step(global_step, loss)
                        if (self.cfg.save_every > 0 and self.cfg.save_dir
                                and global_step > 0 and global_step % self.cfg.save_every == 0):
                            self.save_checkpoint(self.cfg.save_dir)
                        global_step += 1
        finally:
            logger.finish()
        return {"losses": losses, "device": str(self.device)}
