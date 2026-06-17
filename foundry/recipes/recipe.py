"""
Recipe — load a YAML recipe, inspect it with plan(), execute with run().
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional, Union

from foundry.recipes.schema import EmbedRecipe, FoundryRecipe
from foundry.contracts import ArchConfig
from foundry.backends import detect_backend


class Recipe:
    """
    The main entry point for the factory.

    Usage::

        recipe = Recipe.load("ola_15b.yaml")
        for line in recipe.plan():
            print(line)

        base = recipe.run()   # requires torch backend + GPU
    """

    def __init__(self, spec: Union[FoundryRecipe, EmbedRecipe]) -> None:
        self._spec = spec

    @classmethod
    def load(cls, path: str | Path) -> "Recipe":
        """Parse and validate a YAML recipe file, auto-detecting recipe type."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load recipes. "
                "Install with: pip install pyyaml"
            )
        raw = yaml.safe_load(Path(path).read_text())

        # EmbedRecipe is identified by embed_loss / embed_pool in fusion block.
        fusion = raw.get("fusion", {})
        is_embed = "embed_loss" in fusion or "embed_pool" in fusion
        if is_embed:
            spec: Union[FoundryRecipe, EmbedRecipe] = EmbedRecipe.model_validate(raw)
        else:
            spec = FoundryRecipe.model_validate(raw)
        return cls(spec)

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        """Build a Recipe from a plain dict (useful in tests)."""
        fusion = d.get("fusion", {})
        is_embed = "embed_loss" in fusion or "embed_pool" in fusion
        if is_embed:
            return cls(EmbedRecipe.model_validate(d))
        return cls(FoundryRecipe.model_validate(d))

    def plan(self) -> list[str]:
        """
        Return a human-readable plan for the recipe — no compute, no GPU.

        Estimates parameter counts, FLOPs, and costs for each stage.
        Highlights shape warnings before any money is spent.
        """
        lines: list[str] = ["=" * 60, "  olaverse-foundry — Recipe Plan", "=" * 60]
        s = self._spec

        if isinstance(s, EmbedRecipe):
            lines.append("\n[Type] Embedding Distillation Recipe")

        # ── Seed ──────────────────────────────────────────────────
        lines.append("\n[1] Seed")
        if s.seed.init == "pretrained":
            lines.append(f"    Warm-start from: {s.seed.model}")
            lines.append("    Estimated seed cost: ~$0 (loading existing model)")
        else:
            lines.append(f"    Custom arch: {s.seed.arch}")
            if s.seed.pretrain:
                tokens = s.seed.pretrain.get("tokens", 0)
                lines.append(f"    Pre-train tokens: {tokens:.2e}")
                lines.append("    Estimated seed cost: $75,000 – $120,000 (novel arch, cluster)")

        # ── Data ──────────────────────────────────────────────────
        if s.data:
            lines.append(f"\n[1b] Data")
            lines.append(f"    Source:  {s.data.source}  split={s.data.split}")
            lines.append(
                f"    batch_size={s.data.batch_size}  "
                f"max_length={s.data.max_length}  "
                f"streaming={s.data.streaming}"
            )

        # ── Grow (FoundryRecipe only) ──────────────────────────────
        if isinstance(s, FoundryRecipe) and s.grow:
            lines.append(f"\n[2] Grow  ({s.grow.method})")
            lines.append(f"    Target size: {s.grow.to_params / 1e9:.1f}B parameters")
            lines.append("    Method: SOLAR-style depth upscale (duplicate-and-trim layers)")
            lines.append("    Cost: ~$0 (no training, just weight copy)")
            lines.append("    ⚠  Model will underperform seed until healed — expected SOLAR dip.")

        # ── Teachers ──────────────────────────────────────────────
        if s.teachers:
            lines.append(f"\n[3] Teachers  ({len(s.teachers)} total)")
            for t in s.teachers:
                lines.append(f"    [{t.role:12s}] {t.model}  weight={t.weight}")
            if isinstance(s, FoundryRecipe):
                lines.append(f"    Alignment: {s.fusion.align}  |  Cache: {s.fusion.cache}")
            lines.append(
                "    ⚠  Teacher inference can rival training cost — cache aggressively."
            )

        # ── Heal ──────────────────────────────────────────────────
        if s.heal:
            tokens = s.heal.tokens
            lines.append(f"\n[4] Heal / Fuse  ({tokens:.2e} tokens)")
            if isinstance(s, FoundryRecipe):
                lines.append(f"    Strategy: {s.fusion.strategy}  |  alpha={s.heal.alpha}")
                lines.append(
                    f"    Loss: {s.heal.alpha:.1f}·CE(student,gold) + "
                    f"{1-s.heal.alpha:.1f}·KL(student ‖ fused_teacher)"
                )
            rough_usd = int(tokens / 1e9 * 2.5)
            lines.append(f"    Estimated cost: ~${rough_usd:,}  (32× H100, on-demand)")

        # ── Output ────────────────────────────────────────────────
        lines.append(f"\n[5] Output")
        lines.append(f"    Freeze base: {s.output.freeze_base}")
        if isinstance(s, FoundryRecipe) and s.output.skillpacks:
            lines.append(f"    Skill packs to train: {', '.join(s.output.skillpacks)}")
            lines.append("    Estimated cost per pack: ~$100 – $500  (1× GPU, hours)")
        if s.output.save_path:
            lines.append(f"    Save to: {s.output.save_path}")

        # ── Backend check ─────────────────────────────────────────
        backend = detect_backend()
        lines.append(f"\n[Backend] {backend['summary']}")
        if not backend["torch"]:
            lines.append(
                "    ⚠  torch not found — run() will fall back to toy backend. "
                "Install with: pip install olaverse-foundry[torch]"
            )

        lines.append("\n" + "=" * 60)
        lines.append("  Run `recipe.run()` to execute (requires GPU backend).")
        lines.append("=" * 60)
        return lines

    def run(
        self,
        backend: str = "auto",
        dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
    ) -> Any:
        """
        Execute the recipe end-to-end.

        Args:
            backend:      "auto" | "toy" | "torch"
            dataset:      Optional training data. Accepts:
                          - A list of np.ndarray token batches (shape B×T)
                          - A list of str (text passages)
                          - A DataPipeline instance
                          - None — recipe must have a data: block in that case
            eval_dataset: Same format as dataset, used for validation loss.
        """
        info = detect_backend()
        if backend == "toy" or (backend == "auto" and not info["torch"]):
            return self._run_toy()

        if isinstance(self._spec, EmbedRecipe):
            return self._run_embed(dataset=dataset, eval_dataset=eval_dataset)
        return self._run_torch(dataset=dataset, eval_dataset=eval_dataset)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_dataset(self, dataset: Optional[Any], s: Any) -> Any:
        """Resolve dataset: explicit arg > recipe data: block > error."""
        if dataset is not None:
            return dataset

        if s.data is not None:
            from foundry.data import DataPipeline
            mode = "embed" if isinstance(s, EmbedRecipe) else "lm"
            dp = DataPipeline(
                source=s.data.source,
                split=s.data.split,
                text_column=s.data.text_column,
                batch_size=s.data.batch_size,
                max_length=s.data.max_length,
                streaming=s.data.streaming,
                shuffle_buffer=s.data.shuffle_buffer or None,
                mode=mode,
            )
            return dp

        return None

    def _run_toy(self) -> dict:
        """M0 toy run — exercises the full pipeline with numpy, no GPU."""
        import numpy as np
        from foundry.teachers import TeacherRegistry
        from foundry.training import DistillTrainer, TrainConfig

        vocab = 128
        seq   = 16
        n_teachers = max(1, len(self._spec.teachers))

        class _ToyStudent:
            config = ArchConfig(n_layers=2, d_model=64, vocab_size=vocab, name="toy")
            def forward(self, ids): return np.random.randn(*ids.shape, vocab).astype(np.float32)
            def parameters(self): return []
            tokenizer = None

        data = [np.random.randint(0, vocab, (2, seq)) for _ in range(4)]
        teachers = TeacherRegistry.from_toy(
            n=n_teachers, vocab_size=vocab,
            weights=[t.weight for t in self._spec.teachers] or None,
        )
        cfg = TrainConfig(
            epochs=1,
            alpha=self._spec.heal.alpha if self._spec.heal else 0.3,
            fusion_strategy=self._spec.fusion.strategy if isinstance(self._spec, FoundryRecipe) else "min_ce",
        )
        history = DistillTrainer(_ToyStudent(), teachers, cfg).train(data)
        return {"mode": "toy", "final_loss": history["losses"][-1]}

    def _run_torch(
        self,
        dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
    ) -> dict:
        """
        Full torch run: seed → optional grow → teacher align → heal/distill → output.
        """
        from foundry.io import load_seed
        from foundry.teachers import TeacherRegistry
        from foundry.training import TorchDistillTrainer, TorchTrainConfig

        s = self._spec
        assert isinstance(s, FoundryRecipe)

        # ── 1. Load seed ───────────────────────────────────────────
        print("[foundry] Loading seed model …")
        seed = load_seed(s.seed)
        model = seed.model
        tokenizer = seed.tokenizer
        print(f"[foundry] Seed loaded: {seed.model_id!r} ({seed.strategy})")

        # ── 2. Grow (depth upscale) ────────────────────────────────
        if s.grow:
            from foundry.growth import plan_growth, build_upscaled_state_dict
            print(f"[foundry] Growing to {s.grow.to_params/1e9:.1f}B via {s.grow.method} …")
            growth_plan = plan_growth(model, target_params=s.grow.to_params)
            new_state = build_upscaled_state_dict(
                model.state_dict(), growth_plan.layer_map
            )
            model.load_state_dict(new_state, strict=False)
            print(f"[foundry] Grow complete. New layer count: {growth_plan.n_layers_new}")

        # ── 3. Build teacher pool ──────────────────────────────────
        names   = [t.model for t in s.teachers]
        weights = [t.weight for t in s.teachers]
        if names:
            print(f"[foundry] Loading {len(names)} teacher(s) …")
            teachers = TeacherRegistry.from_names(names, weights=weights)
            teachers.load_all(device="auto")
        else:
            teachers = TeacherRegistry.from_toy(n=1, vocab_size=getattr(seed.config, "vocab_size", 32000))
            warnings.warn(
                "No teachers specified — using a toy teacher for the heal phase.",
                UserWarning, stacklevel=2,
            )

        # ── 4. Resolve training data ───────────────────────────────
        train_data = self._build_dataset(dataset, s)
        if train_data is None:
            # Fallback: synthesise batches matching token budget (still useful for testing)
            import numpy as np
            heal_tokens = s.heal.tokens if s.heal else int(1e6)
            batch_size  = 4
            seq_len     = 512
            n_batches   = max(1, int(heal_tokens / (batch_size * seq_len)))
            vocab_size  = getattr(seed.config, "vocab_size", 32000)
            train_data  = [
                np.random.randint(0, vocab_size, (batch_size, seq_len))
                for _ in range(n_batches)
            ]
            warnings.warn(
                "No dataset provided and recipe has no data: block — "
                "training on synthetic random tokens. "
                "Pass dataset= or add a data: block to your recipe YAML.",
                UserWarning, stacklevel=3,
            )

        eval_data = self._build_dataset(eval_dataset, s) if eval_dataset is None else eval_dataset

        # ── 5. Freeze base weights if requested ───────────────────
        if s.output.freeze_base and s.output.skillpacks:
            print("[foundry] Freezing base model weights (output.freeze_base=true) …")
            for param in model.parameters():
                param.requires_grad_(False)

        # ── 6. Heal / distill ──────────────────────────────────────
        alpha = s.heal.alpha if s.heal else 0.3
        cfg = TorchTrainConfig(
            epochs=1,
            alpha=alpha,
            fusion_strategy=s.fusion.strategy,
        )
        trainer = TorchDistillTrainer(model, teachers, config=cfg)

        def _log(step: int, loss: float) -> None:
            if step % 100 == 0:
                print(f"[foundry]   step {step:>6}  loss={loss:.4f}")

        history = trainer.train(train_data, on_step=_log, eval_data=eval_data)
        print(f"[foundry] Training complete. Final loss: {history['losses'][-1]:.4f}")

        # ── 7. Skill packs ─────────────────────────────────────────
        if s.output.skillpacks:
            warnings.warn(
                f"Skill packs {s.output.skillpacks} are declared but recipe.run() does not "
                "train them automatically yet. Load them via SkillRegistry.load() after saving.",
                UserWarning, stacklevel=2,
            )

        # ── 8. Save ────────────────────────────────────────────────
        if s.output.save_path:
            save_dir = Path(s.output.save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(save_dir))
            if tokenizer is not None:
                tokenizer.save_pretrained(str(save_dir))
            print(f"[foundry] Model saved to: {save_dir}")

        return {
            "mode":       "torch",
            "device":     history.get("device", "unknown"),
            "final_loss": history["losses"][-1],
            "seed":       seed.model_id,
            "save_path":  s.output.save_path,
        }

    def _run_embed(
        self,
        dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
    ) -> dict:
        """
        Embedding distillation run: student encoder ← pooled teacher embeddings.
        """
        from transformers import AutoModel, AutoTokenizer

        from foundry.training import EmbeddingDistillTrainer, EmbeddingDistillConfig

        s = self._spec
        assert isinstance(s, EmbedRecipe)

        if not s.teachers:
            raise ValueError("EmbedRecipe requires at least one teacher.")

        # ── Load student ───────────────────────────────────────────
        print(f"[foundry] Loading student encoder: {s.seed.model} …")
        student   = AutoModel.from_pretrained(s.seed.model)
        tokenizer = AutoTokenizer.from_pretrained(s.seed.model)

        # ── Load teacher(s) ───────────────────────────────────────
        # For embedding distillation we load the first teacher as the canonical
        # target; additional teachers are averaged at the embedding level.
        teacher_models = []
        for spec in s.teachers:
            print(f"[foundry] Loading teacher encoder: {spec.model} …")
            teacher_models.append(AutoModel.from_pretrained(spec.model))

        if len(teacher_models) > 1:
            warnings.warn(
                f"EmbedRecipe has {len(teacher_models)} teachers. "
                "Their embeddings will be averaged during training.",
                UserWarning, stacklevel=2,
            )

        # ── Resolve training data ──────────────────────────────────
        train_data = self._build_dataset(dataset, s)
        if train_data is None:
            raise ValueError(
                "EmbedRecipe.run() requires training data. "
                "Either pass dataset= or add a data: block to your recipe YAML.\n"
                "Example:\n"
                "  data:\n"
                "    source: sentence-transformers/natural-questions\n"
                "    split: train\n"
                "    text_column: query"
            )

        eval_data = self._build_dataset(eval_dataset, s) if eval_dataset is None else eval_dataset

        # ── Build config ───────────────────────────────────────────
        teacher_dim = teacher_models[0].config.hidden_size
        student_dim = student.config.hidden_size

        cfg = EmbeddingDistillConfig(
            loss=s.fusion.embed_loss,
            pool=s.fusion.embed_pool,
            normalize=s.fusion.normalize,
            project_dim=teacher_dim if student_dim != teacher_dim else 0,
            epochs=1,
            alpha=s.heal.alpha if s.heal else 0.0,
        )

        # ── Train ──────────────────────────────────────────────────
        trainer = EmbeddingDistillTrainer(
            student=student,
            teachers=teacher_models,
            tokenizer=tokenizer,
            config=cfg,
        )
        history = trainer.train(train_data, eval_data=eval_data)
        print(f"[foundry] Embedding distillation complete. Final loss: {history['losses'][-1]:.4f}")

        # ── Save ───────────────────────────────────────────────────
        if s.output.save_path:
            save_dir = Path(s.output.save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            student.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            print(f"[foundry] Student encoder saved to: {save_dir}")

        return {
            "mode":       "embed",
            "final_loss": history["losses"][-1],
            "student":    s.seed.model,
            "teacher":    s.teachers[0].model,
            "save_path":  s.output.save_path,
        }
