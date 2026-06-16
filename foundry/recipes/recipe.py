"""
Recipe — load a YAML recipe, inspect it with plan(), execute with run().
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from foundry.recipes.schema import FoundryRecipe
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

    def __init__(self, spec: FoundryRecipe) -> None:
        self._spec = spec

    @classmethod
    def load(cls, path: str | Path) -> "Recipe":
        """Parse and validate a YAML recipe file."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load recipes. "
                "Install with: pip install olaverse-foundry"
            )
        raw = yaml.safe_load(Path(path).read_text())
        spec = FoundryRecipe.model_validate(raw)
        return cls(spec)

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        """Build a Recipe from a plain dict (useful in tests)."""
        return cls(FoundryRecipe.model_validate(d))

    def plan(self) -> list[str]:
        """
        Return a human-readable plan for the recipe — no compute, no GPU.

        Estimates parameter counts, FLOPs, and costs for each stage.
        Highlights shape warnings before any money is spent.
        """
        lines: list[str] = ["=" * 60, "  olaverse-foundry — Recipe Plan", "=" * 60]
        s = self._spec

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

        # ── Grow ──────────────────────────────────────────────────
        if s.grow:
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
            lines.append(f"    Alignment: {s.fusion.align}  |  Cache: {s.fusion.cache}")
            lines.append(
                "    ⚠  Teacher inference can rival training cost — cache aggressively."
            )

        # ── Heal ──────────────────────────────────────────────────
        if s.heal:
            tokens = s.heal.tokens
            lines.append(f"\n[4] Heal / Fuse  ({tokens:.2e} tokens)")
            lines.append(f"    Strategy: {s.fusion.strategy}  |  alpha={s.heal.alpha}")
            lines.append(
                f"    Loss: {s.heal.alpha:.1f}·CE(student,gold) + "
                f"{1-s.heal.alpha:.1f}·KL(student ‖ fused_teacher)"
            )
            # Rough cost estimate (assumes 15B-class model, 32 H100s)
            rough_usd = int(tokens / 1e9 * 2.5)
            lines.append(f"    Estimated cost: ~${rough_usd:,}  (32× H100, on-demand)")

        # ── Output ────────────────────────────────────────────────
        lines.append(f"\n[5] Output")
        lines.append(f"    Freeze base: {s.output.freeze_base}")
        if s.output.skillpacks:
            lines.append(f"    Skill packs to train: {', '.join(s.output.skillpacks)}")
            lines.append("    Estimated cost per pack: ~$100 – $500  (1× GPU, hours)")
        if s.output.save_path:
            lines.append(f"    Save to: {s.output.save_path}")

        # ── Backend check ─────────────────────────────────────────
        backend = detect_backend()
        lines.append(f"\n[Backend] {backend['summary']}")
        if not backend["torch"]:
            lines.append(
                "    ⚠  torch not found — run() will fail. "
                "Install with: pip install olaverse-foundry[torch]"
            )

        lines.append("\n" + "=" * 60)
        lines.append("  Run `recipe.run()` to execute (requires GPU backend).")
        lines.append("=" * 60)
        return lines

    def run(self, backend: str = "auto") -> Any:
        """
        Execute the recipe end-to-end.

        Args:
            backend: "auto" (default) — use torch if available, toy otherwise.
                     "toy"  — always run the numpy toy backend (no GPU, CI-safe).
                     "torch" — require the torch backend (raises if not installed).

        M0: toy path exercises the full pipeline without a GPU.
        M3: torch path dispatches to the real distillation trainer.
        """
        info = detect_backend()
        if backend == "toy" or (backend == "auto" and not info["torch"]):
            return self._run_toy()
        return self._run_torch()

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

        dataset = [np.random.randint(0, vocab, (2, seq)) for _ in range(4)]
        teachers = TeacherRegistry.from_toy(n=n_teachers, vocab_size=vocab,
                                            weights=[t.weight for t in self._spec.teachers] or None)
        cfg = TrainConfig(
            epochs=1,
            alpha=self._spec.heal.alpha if self._spec.heal else 0.3,
            fusion_strategy=self._spec.fusion.strategy,
        )
        history = DistillTrainer(_ToyStudent(), teachers, cfg).train(dataset)
        return {"mode": "toy", "final_loss": history["losses"][-1]}

    def _run_torch(self) -> Any:
        """M3 torch distillation run — not yet implemented."""
        raise NotImplementedError(
            "Torch-backed training lands in M3. "
            "Use recipe.run() with the toy backend for now, "
            "or implement foundry/training/torch_distill.py."
        )
