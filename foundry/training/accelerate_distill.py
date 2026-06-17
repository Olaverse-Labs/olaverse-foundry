"""
CachedDistillTrainer — distillation trainer with on-disk LogitCache + accelerate.

Full production feature set:
  • On-disk LogitCache — teachers run once; subsequent epochs free.
  • accelerate.Accelerator — transparent DDP/FSDP + gradient accumulation.
  • Mixed precision via torch.autocast.
  • LR scheduler: "constant" | "cosine" | "linear" with warmup.
  • Reproducible seed.
  • Dataset shuffling.
  • Checkpoint save / resume (model + optimizer + caches).
  • OOM error with actionable message.
  • Optional W&B / TensorBoard logging.
  • Accepts any iterable dataset.
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
from foundry.training.torch_distill import TorchTrainConfig
from foundry.training._logger import _FoundryLogger
from foundry.training._scheduler import build_scheduler
from foundry.teachers.cache import LogitCache


@dataclass
class CachedDistillConfig(TorchTrainConfig):
    """
    Extends TorchTrainConfig with caching and accelerate options.

    Extra fields:
        cache_dir:      Directory to save/load on-disk logit caches. None = memory only.
        cache_top_k:    Top-k values to cache per position.
        use_accelerate: Try to init accelerate.Accelerator (DDP/FSDP). Falls back
                        to plain torch if accelerate is not installed.
    """
    cache_dir:      Optional[str] = None
    cache_top_k:    int           = 64
    use_accelerate: bool          = True


class CachedDistillTrainer:
    """
    Distillation trainer with on-disk LogitCache and accelerate support.

    Cache strategy:
      • First pass: teachers run live and results are stored (memory + disk if
        ``cache_dir`` is set).
      • Subsequent epochs: read exclusively from cache — zero teacher cost.

    Args:
        student:   ``nn.Module`` whose forward returns ``.logits``.
        teachers:  ``TeacherRegistry`` — teachers must already be loaded if real.
        config:    ``CachedDistillConfig``.
        alignment: Tokenizer alignment. Defaults to ``IdentityAlignment``.
    """

    def __init__(
        self,
        student,
        teachers,
        config:    CachedDistillConfig | None = None,
        alignment = None,
    ) -> None:
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is required for CachedDistillTrainer. "
                "Install with: pip install olaverse-foundry[torch]"
            )
        self.student    = student
        self.teachers   = teachers
        self.cfg        = config or CachedDistillConfig()
        self._alignment = alignment
        self._caches: list[LogitCache] = [
            LogitCache(top_k=self.cfg.cache_top_k) for _ in self.teachers
        ]
        self._accelerator = self._init_accelerator()
        self.device       = self._resolve_device()
        self._dtype       = self._resolve_dtype()
        self.student.to(self.device)
        self._optimizer   = self._build_optimizer()

    # ── Setup ───────────────────────────────────────────────────────────────

    def _init_accelerator(self):
        if not self.cfg.use_accelerate:
            return None
        try:
            from accelerate import Accelerator
            return Accelerator(
                gradient_accumulation_steps=self.cfg.grad_accumulation_steps
            )
        except ImportError:
            return None

    def _resolve_device(self):
        import torch
        if self._accelerator is not None:
            return self._accelerator.device
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
        """Save student weights, optimizer, and in-memory caches."""
        import torch
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        ckpt = p / "checkpoint.pt"
        student = getattr(self.student, "module", self.student)
        torch.save({
            "model_state":     student.state_dict(),
            "optimizer_state": self._optimizer.state_dict(),
            "config":          vars(self.cfg),
        }, ckpt)
        for i, cache in enumerate(self._caches):
            cache.save(p / f"cache_teacher_{i}.npz")
        return ckpt

    def resume_from_checkpoint(self, path: str | Path) -> None:
        """Load student weights, optimizer, and any saved caches."""
        import torch
        p = Path(path)
        ckpt = p if p.suffix == ".pt" else p / "checkpoint.pt"
        data = torch.load(ckpt, map_location=self.device, weights_only=False)
        student = getattr(self.student, "module", self.student)
        student.load_state_dict(data["model_state"])
        self._optimizer.load_state_dict(data["optimizer_state"])
        for i, cache in enumerate(self._caches):
            cp = p / f"cache_teacher_{i}.npz"
            if cp.exists():
                cache.load(cp)

    # ── Cache management ─────────────────────────────────────────────────────

    def _cache_path(self, teacher_idx: int) -> Optional[Path]:
        if not self.cfg.cache_dir:
            return None
        p = Path(self.cfg.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / f"teacher_{teacher_idx}.npz"

    def load_caches(self) -> bool:
        if not self.cfg.cache_dir:
            return False
        loaded = 0
        for i, cache in enumerate(self._caches):
            path = self._cache_path(i)
            if path and path.exists():
                cache.load(path)
                loaded += 1
        return loaded == len(self._caches)

    def build_caches(self, dataset) -> None:
        """Run each teacher once over the dataset and cache results."""
        dataset_list = list(dataset)
        for i, teacher in enumerate(self.teachers):
            name = getattr(teacher, "name", f"teacher_{i}")
            print(f"[foundry] Caching {name} over {len(dataset_list)} batches …")
            self._caches[i].populate_dataset(teacher, dataset_list)
            path = self._cache_path(i)
            if path:
                self._caches[i].save(path)
                print(f"[foundry]   saved → {path}")

    # ── Training step ────────────────────────────────────────────────────────

    def train_step(self, batch_idx: int, input_ids: np.ndarray) -> float:
        import torch
        import torch.nn.functional as F
        from foundry.fusion.align import IdentityAlignment
        from foundry.fusion.strategies import STRATEGY_REGISTRY

        ids_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
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
                    for i, teacher in enumerate(self.teachers):
                        cached = self._caches[i].get_batch(batch_idx)
                        if cached is None:
                            t_idx, t_prob = teacher.distribution(
                                input_ids, top_k=self.cfg.cache_top_k
                            )
                            self._caches[i].put_batch(batch_idx, t_idx, t_prob)
                        else:
                            t_idx, t_prob = cached
                        teacher_dists.append(align.map(t_idx, t_prob, V))
                        teacher_weights.append(teacher.weight)

                    gold_np  = np.roll(input_ids, -1, axis=1)
                    gold_np[:, -1] = 0
                    fused_np = fuse(teacher_dists, gold_np, teacher_weights)
                    fused_t  = torch.tensor(fused_np, dtype=torch.float32, device=self.device)
                    fused_t  = (fused_t + 1e-9) / (fused_t + 1e-9).sum(dim=-1, keepdim=True)

                    log_stu = F.log_softmax(student_logits[:, :-1], dim=-1)   # (B, S-1, V)
                    # Per-token KL in nats, masking pad positions, so it sits on the
                    # same scale as the per-token CE and alpha truly balances the two.
                    kl_per_tok = F.kl_div(
                        log_stu, fused_t[:, :-1],
                        reduction="none", log_target=False,
                    ).sum(dim=-1)                                             # (B, S-1)
                    kl_mask = (gold_t[:, :-1] != 0).float()
                    kl_loss = (kl_per_tok * kl_mask).sum() / kl_mask.sum().clamp(min=1.0)

                loss = self.cfg.alpha * ce_loss + (1.0 - self.cfg.alpha) * kl_loss

            if self._accelerator is not None:
                with self._accelerator.accumulate(self.student):
                    self._accelerator.backward(loss)
                    if self._accelerator.sync_gradients:
                        self._accelerator.clip_grad_norm_(
                            self.student.parameters(), self.cfg.max_grad_norm
                        )
                    self._optimizer.step()
                    self._optimizer.zero_grad()
            else:
                loss.backward()
                import torch.nn as nn
                nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.max_grad_norm)
                self._optimizer.step()

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                import torch as _t
                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
                self._optimizer.zero_grad()
                raise RuntimeError(
                    "CUDA out of memory. Suggestions:\n"
                    "  • Reduce batch size\n"
                    "  • Increase grad_accumulation_steps\n"
                    "  • Set torch_dtype='bfloat16'\n"
                    f"Original: {exc}"
                ) from exc
            raise

        return float(loss.item())

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
        Run the full distillation loop with caching.

        Args:
            dataset:      Iterable of (B, S) int arrays or a DataPipeline.
            eval_dataset: Optional eval iterable.
            on_step:      Callback(step, loss) fired after each optimizer step.
            shuffle:      Shuffle dataset each epoch (finite datasets only).
            total_steps:  Override total optimizer steps for LR scheduler.

        Returns:
            dict with "losses", "eval_losses", "device", "cache_stats".
        """
        self._seed_everything()

        if self._accelerator is not None:
            self.student, self._optimizer = self._accelerator.prepare(
                self.student, self._optimizer
            )

        # Pre-build caches (materialise streaming to list once)
        if not self.load_caches():
            self.build_caches(dataset)

        # dataset_list holds what we'll iterate over each epoch
        try:
            dataset_list = list(dataset)
        except Exception:
            dataset_list = dataset

        if total_steps is None:
            try:
                n_batches   = len(dataset_list)  # type: ignore[arg-type]
                total_steps = math.ceil(
                    n_batches * self.cfg.epochs
                    / max(1, self.cfg.grad_accumulation_steps)
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
        step = 0

        try:
            for epoch in range(self.cfg.epochs):
                indices = list(range(len(dataset_list)))
                if shuffle:
                    random.shuffle(indices)

                for batch_idx in indices:
                    batch_ids = dataset_list[batch_idx]
                    if isinstance(batch_ids, dict):
                        batch_ids = batch_ids["input_ids"]
                    if not isinstance(batch_ids, np.ndarray):
                        batch_ids = np.array(batch_ids)
                    if batch_ids.ndim == 1:
                        batch_ids = batch_ids[None, :]

                    loss = self.train_step(batch_idx, batch_ids)
                    losses.append(loss)

                    if scheduler is not None:
                        scheduler.step()
                    logger.log(step, loss)
                    if on_step and step % self.cfg.log_every == 0:
                        on_step(step, loss)

                    if (eval_dataset is not None
                            and self.cfg.eval_every > 0
                            and step % self.cfg.eval_every == 0):
                        ev = self._run_eval(eval_dataset)
                        eval_losses[step] = ev
                        logger.log_eval(step, ev)

                    if (self.cfg.save_every > 0
                            and self.cfg.save_dir
                            and step > 0
                            and step % self.cfg.save_every == 0):
                        self.save_checkpoint(self.cfg.save_dir)

                    step += 1
        finally:
            logger.finish()

        return {
            "losses":      losses,
            "eval_losses": eval_losses,
            "device":      str(self.device),
            "cache_stats": [c.stats for c in self._caches],
        }

    def _run_eval(self, eval_dataset) -> float:
        import torch
        import torch.nn.functional as F
        student = getattr(self.student, "module", self.student)
        student.eval()
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
                out    = student(input_ids=ids_t)
                logits = out.logits.float()
                V      = logits.shape[-1]
                gold_t = torch.roll(ids_t, -1, dims=1)
                gold_t[:, -1] = 0
                loss   = F.cross_entropy(
                    logits[:, :-1].contiguous().view(-1, V),
                    gold_t[:, :-1].contiguous().view(-1),
                    ignore_index=0,
                )
                total += float(loss.item())
                n     += 1
        student.train()
        return total / max(1, n)
