"""
EmbeddingDistillTrainer — sentence-embedding distillation.

Full production feature set:
  • Teacher embedding cache (build_embed_cache / pre_cache=True).
  • Mixed precision via torch.autocast.
  • Gradient accumulation.
  • LR scheduler: "constant" | "cosine" | "linear" with warmup.
  • Reproducible seed.
  • Dataset shuffling.
  • Eval loop (eval_every, returns eval_losses per step).
  • Auto-checkpoint every N steps (save_every, save_dir).
  • Checkpoint save / resume.
  • OOM error with actionable message.
  • Optional W&B / TensorBoard logging.
  • Accepts any iterable dataset (list, DataPipeline, HF IterableDataset).
"""
from __future__ import annotations

import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from foundry.training._logger import _FoundryLogger
from foundry.training._scheduler import build_scheduler


# ── Pooling helpers ────────────────────────────────────────────────────────

def _pool(hidden_states, attention_mask, mode: str):
    """Pool (B, S, D) hidden states to (B, D) sentence vector."""
    import torch
    if mode == "cls":
        return hidden_states[:, 0, :]
    if attention_mask is None:
        return hidden_states.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingDistillConfig:
    """
    Configuration for embedding distillation.

    Core hyperparameters
    --------------------
    loss:           "mse" | "cosine" on pooled sentence vectors.
    pool:           "mean" (recommended) | "cls".
    normalize:      L2-normalise embeddings before computing loss.
    temperature:    Divide embeddings by this before loss (> 1 sharpens).
    project_dim:    If > 0, add a trainable linear projection (student_dim → project_dim)
                    before computing the loss. Required when student_dim ≠ teacher_dim.
                    Set to 0 to disable (default; raises if dims mismatch instead).
    alpha:          Not used during embedding training (kept for recipe compatibility).
    learning_rate:  AdamW learning rate.
    epochs:         Training epochs.
    weight_decay:   AdamW weight decay.
    max_grad_norm:  Gradient clipping norm.
    device:         "auto" | "cuda" | "mps" | "cpu".

    LR scheduling
    -------------
    lr_scheduler:   "constant" | "cosine" | "linear".
    warmup_steps:   Linear warmup before scheduler kicks in.

    Accumulation & precision
    ------------------------
    grad_accumulation_steps: Accumulate gradients over N batches.
    torch_dtype:    "float32" | "bfloat16" | "float16".

    Evaluation & checkpointing
    --------------------------
    eval_every:     Evaluate every N optimizer steps. 0 = disabled.
    save_every:     Auto-save checkpoint every N steps. 0 = disabled.
    save_dir:       Directory for auto-saved checkpoints.

    Logging
    -------
    log_every:      Trigger on_step callback every N optimizer steps.
    log_backend:    "none" | "wandb" | "tensorboard".
    run_name:       Human-readable run label.
    project:        W&B project name.

    Reproducibility
    ---------------
    seed:           Set torch + numpy + random seed before training. 0 = no seed.
    """
    loss:                    str   = "mse"
    pool:                    str   = "mean"
    normalize:               bool  = True
    temperature:             float = 1.0
    project_dim:             int   = 0
    alpha:                   float = 0.0
    learning_rate:           float = 2e-5
    epochs:                  int   = 3
    weight_decay:            float = 0.01
    max_grad_norm:           float = 1.0
    device:                  str   = "auto"
    lr_scheduler:            str   = "constant"
    warmup_steps:            int   = 0
    grad_accumulation_steps: int   = 1
    torch_dtype:             str   = "float32"
    eval_every:              int   = 0
    save_every:              int   = 0
    save_dir:                str   = ""
    log_every:               int   = 50
    log_backend:             str   = "none"
    run_name:                str   = ""
    project:                 str   = "olaverse-foundry"
    seed:                    int   = 42


# ── Trainer ────────────────────────────────────────────────────────────────

class EmbeddingDistillTrainer:
    """
    Train a compact sentence embedding student by mimicking a larger teacher.

    The student must produce ``last_hidden_state`` (B, S, D) when called as
    ``student(input_ids=..., attention_mask=...)``.

    The teacher interface is flexible:
      • ``.encode(input_ids, attention_mask) → np.ndarray`` callable.
      • HF encoder model — pooled the same way as the student.
      • ``ToyEmbeddingTeacher`` for tests (no download required).
      • A list of the above — embeddings are averaged across teachers.

    Dimension mismatch:
      If student_dim ≠ teacher_dim, set ``config.project_dim = teacher_dim`` to
      add a trainable linear projector (student_dim → teacher_dim). Without this,
      a clear ``ValueError`` is raised at the first forward pass.

    For large teachers, pre-compute embeddings once via ``build_embed_cache()``
    or pass ``pre_cache=True`` to ``train()``.

    Accepts any iterable dataset: ``list[dict]``, ``DataPipeline``, HF Dataset.

    Args:
        student:    nn.Module returning ``.last_hidden_state`` of shape (B, S, D).
        teacher:    Single teacher or list of teachers. See description above.
        tokenizer:  Optional tokenizer (used when dataset contains raw strings).
        config:     ``EmbeddingDistillConfig``.
    """

    def __init__(
        self,
        student,
        teacher,
        tokenizer=None,
        config:  EmbeddingDistillConfig | None = None,
    ) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "torch is required for EmbeddingDistillTrainer. "
                "Install with: pip install olaverse-foundry[torch]"
            )
        self.student   = student
        # Normalise to list
        self.teachers  = teacher if isinstance(teacher, list) else [teacher]
        self.tokenizer = tokenizer
        self.cfg       = config or EmbeddingDistillConfig()
        self.device    = self._resolve_device()
        self._dtype    = self._resolve_dtype()
        self.student.to(self.device)
        for t in self.teachers:
            if hasattr(t, "to"):
                t.to(self.device)

        # Lazy projector — created on first forward if needed
        self._projector: Any = None
        if self.cfg.project_dim > 0:
            student_dim       = self.student.config.hidden_size
            self._projector   = nn.Linear(student_dim, self.cfg.project_dim, bias=False)
            self._projector.to(self.device)

        self._optimizer   = self._build_optimizer()
        self._embed_cache: dict[int, np.ndarray] = {}

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

    # ── Checkpoint ──────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save student weights + projector + optimizer state to ``<path>/checkpoint.pt``."""
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
        """Load student weights + projector + optimizer state from a checkpoint."""
        import torch
        p = Path(path)
        ckpt = p if p.suffix == ".pt" else p / "checkpoint.pt"
        data = torch.load(ckpt, map_location=self.device, weights_only=True)
        self.student.load_state_dict(data["model_state"])
        self._optimizer.load_state_dict(data["optimizer_state"])
        if self._projector is not None and "projector_state" in data:
            self._projector.load_state_dict(data["projector_state"])

    # ── Teacher embedding cache ──────────────────────────────────────────────

    def build_embed_cache(self, dataset) -> None:
        """
        Run teacher over the full dataset once and cache all embeddings in memory.

        After this, ``train()`` reads from cache instead of calling the teacher
        each step. Recommended for large teachers. For streaming datasets,
        materialises the full dataset into memory once.
        """
        import torch
        items = list(dataset)
        print(f"[foundry] Pre-computing teacher embeddings for {len(items)} batches …")
        for i, batch in enumerate(items):
            if isinstance(batch, np.ndarray):
                batch = {"input_ids": batch}
            ids_np  = batch["input_ids"]
            mask_np = batch.get("attention_mask")
            ids_t  = torch.tensor(ids_np,  dtype=torch.long, device=self.device)
            mask_t = (
                torch.tensor(mask_np, dtype=torch.long, device=self.device)
                if mask_np is not None else torch.ones_like(ids_t)
            )
            emb = self._teacher_embed(ids_t, mask_t)
            self._embed_cache[i] = emb.cpu().numpy()
        print(f"[foundry] Cached {len(self._embed_cache)} teacher embeddings.")

    # ── Embedding helpers ────────────────────────────────────────────────────

    def _student_embed(self, input_ids, attention_mask):
        import torch.nn.functional as F
        out = self.student(input_ids=input_ids, attention_mask=attention_mask)
        emb = _pool(out.last_hidden_state, attention_mask, self.cfg.pool)
        if self._projector is not None:
            emb = self._projector(emb)
        emb = emb / self.cfg.temperature
        if self.cfg.normalize:
            emb = F.normalize(emb, dim=-1)
        return emb

    def _single_teacher_embed(self, teacher, input_ids, attention_mask):
        import torch
        import torch.nn.functional as F
        if hasattr(teacher, "encode"):
            np_emb = teacher.encode(
                input_ids.cpu().numpy(),
                attention_mask.cpu().numpy() if attention_mask is not None else None,
            )
            emb = torch.tensor(np_emb, dtype=torch.float32, device=self.device)
        elif callable(teacher):
            out = teacher(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(out, "last_hidden_state"):
                emb = _pool(out.last_hidden_state, attention_mask, self.cfg.pool)
            else:
                emb = out
        else:
            raise TypeError(
                f"teacher must have .encode() or be a callable nn.Module, "
                f"got {type(teacher)}"
            )
        emb = emb / self.cfg.temperature
        if self.cfg.normalize:
            emb = F.normalize(emb, dim=-1)
        return emb

    def _teacher_embed(self, input_ids, attention_mask):
        import torch
        with torch.no_grad():
            embs = [
                self._single_teacher_embed(t, input_ids, attention_mask)
                for t in self.teachers
            ]
            return torch.stack(embs, dim=0).mean(dim=0)

    def _compute_loss(self, student_emb, teacher_emb):
        import torch.nn.functional as F
        s_dim = student_emb.shape[-1]
        t_dim = teacher_emb.shape[-1]
        if s_dim != t_dim:
            raise ValueError(
                f"Embedding dimension mismatch: student produces {s_dim}-dim vectors "
                f"but teacher produces {t_dim}-dim vectors. "
                f"Set config.project_dim={t_dim} to add a trainable projector, or "
                "use student and teacher models with the same hidden size."
            )
        if self.cfg.loss == "cosine":
            return (1.0 - F.cosine_similarity(student_emb, teacher_emb, dim=-1)).mean()
        return F.mse_loss(student_emb, teacher_emb)

    def _run_eval(self, eval_dataset) -> float:
        import torch
        import torch.nn.functional as F
        self.student.eval()
        total, n = 0.0, 0
        with torch.no_grad(), self._autocast():
            for batch in eval_dataset:
                if isinstance(batch, np.ndarray):
                    batch = {"input_ids": batch}
                ids_np  = batch["input_ids"]
                mask_np = batch.get("attention_mask")
                ids_t  = torch.tensor(ids_np,  dtype=torch.long, device=self.device)
                mask_t = (
                    torch.tensor(mask_np, dtype=torch.long, device=self.device)
                    if mask_np is not None else torch.ones_like(ids_t)
                )
                s_emb = self._student_embed(ids_t, mask_t)
                t_emb = self._teacher_embed(ids_t, mask_t)
                total += float(self._compute_loss(s_emb, t_emb).item())
                n += 1
        self.student.train()
        return total / max(1, n)

    # ── Training step ────────────────────────────────────────────────────────

    def train_step(
        self,
        batch:              dict,
        cached_teacher_emb: Optional[np.ndarray] = None,
        *,
        is_first_accum: bool = True,
        is_last_accum:  bool = True,
    ) -> float:
        """
        One forward + backward + (conditional) optimizer step.

        Args:
            batch:              dict with "input_ids" (+optional "attention_mask").
            cached_teacher_emb: Pre-computed (B, D) numpy array. Skips teacher.
            is_first_accum:     Zero gradients before this step.
            is_last_accum:      Clip + step after this step.

        Returns:
            Unscaled loss as a Python float.
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        if isinstance(batch, np.ndarray):
            batch = {"input_ids": batch}

        ids_np  = batch["input_ids"]
        mask_np = batch.get("attention_mask")
        n_acc   = max(1, self.cfg.grad_accumulation_steps)

        ids_t  = torch.tensor(ids_np,  dtype=torch.long, device=self.device)
        mask_t = (
            torch.tensor(mask_np, dtype=torch.long, device=self.device)
            if mask_np is not None else torch.ones_like(ids_t)
        )

        if is_first_accum:
            self._optimizer.zero_grad()

        try:
            with self._autocast():
                student_emb = self._student_embed(ids_t, mask_t)

                if cached_teacher_emb is not None:
                    teacher_emb = torch.tensor(
                        cached_teacher_emb, dtype=torch.float32, device=self.device
                    )
                    if self.cfg.normalize:
                        teacher_emb = F.normalize(teacher_emb, dim=-1)
                else:
                    teacher_emb = self._teacher_embed(ids_t, mask_t)

                raw_loss = self._compute_loss(student_emb, teacher_emb)
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
                    "  • Set torch_dtype='bfloat16'\n"
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
        pre_cache: bool = False,
        total_steps: Optional[int] = None,
    ) -> dict:
        """
        Train over the full dataset for ``config.epochs`` epochs.

        Args:
            dataset:      Iterable of dicts (or arrays). Accepts ``DataPipeline``.
            eval_dataset: Optional eval iterable.
            on_step:      Callback(global_step, loss) fired after each optimizer step.
            shuffle:      Shuffle at the start of each epoch.
            pre_cache:    Run teacher over full dataset once before training.
            total_steps:  Override total optimizer step count for LR scheduler.

        Returns:
            dict with "losses", "eval_losses" (dict step→loss), and "device".
        """
        self._seed_everything()

        # Materialise iterable for random access (needed for pre_cache + shuffle)
        try:
            dataset_list = list(dataset)
        except Exception:
            dataset_list = dataset

        if pre_cache and not self._embed_cache:
            self.build_embed_cache(dataset_list)

        n_acc = max(1, self.cfg.grad_accumulation_steps)

        if total_steps is None:
            try:
                n_batches   = len(dataset_list)  # type: ignore[arg-type]
                total_steps = math.ceil(
                    n_batches * self.cfg.epochs / n_acc
                )
            except TypeError:
                total_steps = 0

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

        losses:      list[float]      = []
        eval_losses: dict[int, float] = {}
        global_step: int              = 0
        accum_idx:   int              = 0

        self.student.train()
        self._optimizer.zero_grad()

        try:
            for epoch in range(self.cfg.epochs):
                indices = list(range(len(dataset_list)))
                if shuffle:
                    random.shuffle(indices)

                for i in indices:
                    batch      = dataset_list[i]
                    cached_emb = self._embed_cache.get(i)

                    pos      = accum_idx % n_acc
                    is_first = pos == 0
                    is_last  = pos == n_acc - 1

                    loss = self.train_step(
                        batch,
                        cached_teacher_emb=cached_emb,
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

                        if (eval_dataset is not None
                                and self.cfg.eval_every > 0
                                and global_step % self.cfg.eval_every == 0):
                            ev = self._run_eval(eval_dataset)
                            eval_losses[global_step] = ev
                            logger.log_eval(global_step, ev)

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


# ── Toy teacher for tests ──────────────────────────────────────────────────

class ToyEmbeddingTeacher:
    """
    Deterministic fake embedding teacher — no model download, no GPU.

    Returns random unit vectors seeded on the batch sum so results are
    reproducible within a test.  Used for CI and offline demos.
    """

    def __init__(self, dim: int = 32, seed: int = 42) -> None:
        self.dim  = dim
        self.seed = seed

    def encode(
        self,
        input_ids:      np.ndarray,
        attention_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return (B, dim) unit-normalised float32 embeddings."""
        B     = input_ids.shape[0]
        rng   = np.random.default_rng(self.seed + int(input_ids.sum()))
        raw   = rng.standard_normal((B, self.dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=-1, keepdims=True).clip(min=1e-9)
        return raw / norms
