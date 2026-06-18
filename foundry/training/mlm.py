"""
MLMTrainer — masked-language-modeling pretraining for encoder backbones.

This is the *teacherless, from-scratch* path: a randomly-initialised encoder
(your own architecture, or any HF ``AutoModelForMaskedLM``) learns general
representations directly from raw text by predicting masked tokens. The result
is a **base encoder** you can later attach heads to (embedding, reranker,
classifier, NER, language-ID, …).

Unlike EmbeddingDistillTrainer (which only supervises the pooled sentence
vector), MLM supervises every token position, so the resulting base keeps the
token-level representations that structured-prediction heads (NER) need.

Contract for the student
-------------------------
``student(input_ids=..., attention_mask=...)`` must return an object with a
``.logits`` tensor of shape ``(B, S, vocab_size)`` — i.e. an encoder *with an
MLM head*. Any HF ``AutoModelForMaskedLM`` satisfies this. For a custom encoder
that only returns ``last_hidden_state``, wrap it with :class:`WithMLMHead`.

Full production feature set: mixed precision, gradient accumulation,
cosine/linear/constant LR + warmup, seeded reproducibility, eval loop,
checkpoint save/resume, auto-checkpoint, OOM message, W&B/TensorBoard logging.
Accepts any iterable dataset (``DataPipeline`` in ``mode="embed"``, list of
dicts, list of ``(B, S)`` arrays).
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


# ── MLM head wrapper for custom encoders ────────────────────────────────────

class WithMLMHead:
    """
    Wrap a custom encoder (returns ``.last_hidden_state``) with a tied/untied
    linear MLM head so it returns ``.logits`` for MLMTrainer.

    Example::

        encoder = MyEncoder(...)              # returns .last_hidden_state (B,S,D)
        student = WithMLMHead(encoder, hidden_size=384, vocab_size=32000)
        MLMTrainer(student, tokenizer=tok, config=cfg).train(pipe)
    """

    def __init__(self, encoder, hidden_size: int, vocab_size: int, tie_weights: bool = False):
        import torch.nn as nn

        class _Wrapped(nn.Module):
            def __init__(self, enc):
                super().__init__()
                self.encoder = enc
                self.mlm_head = nn.Linear(hidden_size, vocab_size)
                if tie_weights and hasattr(enc, "embeddings"):
                    # Best-effort weight tying with a token embedding if present.
                    emb = getattr(enc, "get_input_embeddings", lambda: None)()
                    if emb is not None and emb.weight.shape == self.mlm_head.weight.shape:
                        self.mlm_head.weight = emb.weight

            def forward(self, input_ids, attention_mask=None, **kw):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kw)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
                from types import SimpleNamespace
                return SimpleNamespace(logits=self.mlm_head(hidden))

        self._module = _Wrapped(encoder)

    def __getattr__(self, name):
        return getattr(self._module, name)

    def __call__(self, *a, **kw):
        return self._module(*a, **kw)


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class MLMConfig:
    """Configuration for masked-language-modeling pretraining."""

    mask_prob:               float = 0.15
    mask_token_id:           Optional[int] = None   # required if no tokenizer given
    pad_token_id:            int   = 0
    vocab_size:              Optional[int] = None    # inferred from tokenizer if None

    learning_rate:           float = 1e-4
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


class MLMTrainer:
    """Masked-language-modeling pretrainer. See module docstring for the student contract."""

    def __init__(self, student, tokenizer=None, config: MLMConfig | None = None) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "torch is required for MLMTrainer. "
                "Install with: pip install olaverse-foundry[torch]"
            )
        self.student   = student
        self.tokenizer = tokenizer
        self.cfg       = config or MLMConfig()

        # Resolve masking vocabulary from the tokenizer when available.
        mask_id  = self.cfg.mask_token_id
        pad_id   = self.cfg.pad_token_id
        vocab    = self.cfg.vocab_size
        specials = []
        if tokenizer is not None:
            mask_id = mask_id if mask_id is not None else getattr(tokenizer, "mask_token_id", None)
            tpad    = getattr(tokenizer, "pad_token_id", None)
            pad_id  = tpad if tpad is not None else pad_id
            vocab   = vocab if vocab is not None else (len(tokenizer) if hasattr(tokenizer, "__len__") else None)
            specials = list(getattr(tokenizer, "all_special_ids", []) or [])
        if mask_id is None:
            raise ValueError(
                "MLMTrainer needs a mask token id. Pass a tokenizer with a "
                "mask_token, or set MLMConfig.mask_token_id explicitly."
            )
        if vocab is None:
            raise ValueError(
                "MLMTrainer needs vocab_size for random-token replacement. "
                "Pass a tokenizer with __len__, or set MLMConfig.vocab_size."
            )
        self._mask_id  = int(mask_id)
        self._pad_id   = int(pad_id)
        self._vocab    = int(vocab)
        self._specials = set(int(s) for s in specials)

        self.device    = self._resolve_device()
        self._dtype    = self._resolve_dtype()
        self.student.to(self.device)
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

    # ── Masking ───────────────────────────────────────────────────────────────

    def _to_ids_mask(self, batch):
        """Normalise a batch into (input_ids, attention_mask) torch long tensors."""
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
        ids_t  = torch.tensor(ids_np,  dtype=torch.long, device=self.device)
        mask_t = torch.tensor(mask_np, dtype=torch.long, device=self.device)
        return ids_t, mask_t

    def _mask_tokens(self, ids_t, attn_t):
        """
        Apply BERT-style dynamic masking.

        Returns (masked_inputs, labels) where labels are -100 except at the
        positions selected for prediction.
        """
        import torch
        labels = ids_t.clone()
        prob   = torch.full(labels.shape, self.cfg.mask_prob, device=self.device)

        # Never mask special tokens or padding.
        special = torch.zeros_like(ids_t, dtype=torch.bool)
        for sid in self._specials:
            special |= (ids_t == sid)
        special |= (attn_t == 0)
        prob.masked_fill_(special, 0.0)

        masked = torch.bernoulli(prob).bool()

        # Guarantee at least one masked position. If a batch happens to mask
        # nothing (likely for small batch×seq), CrossEntropy with every target ==
        # ignore_index averages over an empty set and returns NaN — which then
        # propagates into the weights and poisons every subsequent step (NaN at
        # any learning rate). Force-mask one random valid (non-special) position.
        if not bool(masked.any()):
            valid = (~special).view(-1).nonzero(as_tuple=False).flatten()
            if valid.numel() > 0:
                pick = valid[torch.randint(valid.numel(), (1,), device=self.device)]
                masked.view(-1)[pick] = True

        labels[~masked] = -100   # only compute loss on masked positions

        inputs = ids_t.clone()
        # 80% -> [MASK]
        repl_mask = torch.bernoulli(torch.full(labels.shape, 0.8, device=self.device)).bool() & masked
        inputs[repl_mask] = self._mask_id
        # 10% -> random token
        rand_mask = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=self.device)).bool()
            & masked & ~repl_mask
        )
        random_tokens = torch.randint(self._vocab, labels.shape, dtype=torch.long, device=self.device)
        inputs[rand_mask] = random_tokens[rand_mask]
        # remaining 10% -> unchanged
        return inputs, labels

    # ── Core step ──────────────────────────────────────────────────────────────

    def train_step(self, batch, *, is_first_accum=True, is_last_accum=True) -> float:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        n_acc = max(1, self.cfg.grad_accumulation_steps)
        if is_first_accum:
            self._optimizer.zero_grad()

        ids_t, attn_t  = self._to_ids_mask(batch)
        inputs, labels = self._mask_tokens(ids_t, attn_t)

        try:
            with self._autocast():
                out    = self.student(input_ids=inputs, attention_mask=attn_t)
                logits = out.logits.float()
                V      = logits.shape[-1]
                loss   = F.cross_entropy(
                    logits.view(-1, V), labels.view(-1), ignore_index=-100
                )
            # Belt-and-suspenders: never backprop a non-finite loss (it would
            # write NaN into the weights and poison the whole run).
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
                    "CUDA out of memory. Suggestions:\n"
                    "  • Reduce batch size or max_length\n"
                    f"  • Increase grad_accumulation_steps (currently {n_acc})\n"
                    "  • Set torch_dtype='bfloat16' in MLMConfig\n"
                    f"Original: {exc}"
                ) from exc
            raise

        if is_last_accum:
            nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.max_grad_norm)
            self._optimizer.step()
        return float(loss.item())

    # ── Eval ────────────────────────────────────────────────────────────────────

    def _run_eval(self, eval_dataset) -> float:
        import torch
        import torch.nn.functional as F
        self.student.eval()
        total, n = 0.0, 0
        with torch.no_grad(), self._autocast():
            for batch in eval_dataset:
                ids_t, attn_t  = self._to_ids_mask(batch)
                inputs, labels = self._mask_tokens(ids_t, attn_t)
                out    = self.student(input_ids=inputs, attention_mask=attn_t)
                logits = out.logits.float()
                V      = logits.shape[-1]
                loss   = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
                total += float(loss.item()); n += 1
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
        """Run MLM pretraining for ``config.epochs`` epochs. Returns
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

        scheduler = build_scheduler(
            self._optimizer, self.cfg.lr_scheduler, self.cfg.warmup_steps, total_steps
        )
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
                nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.max_grad_norm)
                self._optimizer.step()
                self._optimizer.zero_grad()
        finally:
            logger.finish()

        return {"losses": losses, "eval_losses": eval_losses, "device": str(self.device)}
