"""
foundry CLI — plan / run / doctor / strategies / embed

Usage:
    foundry plan       <recipe.yaml>   — show staged plan and cost estimates
    foundry run        <recipe.yaml>   — execute the recipe
    foundry doctor                     — check installed backends and environment
    foundry strategies                 — list available fusion strategies
    foundry embed      <recipe.yaml>   — run an embedding distillation recipe (M5)
"""
from __future__ import annotations

import sys


# ── Subcommand handlers ────────────────────────────────────────────────────

def _plan(recipe_path: str) -> None:
    from foundry.recipes import Recipe
    r = Recipe.load(recipe_path)
    print("\n".join(r.plan()))


def _run(recipe_path: str) -> None:
    from foundry.recipes import Recipe
    r = Recipe.load(recipe_path)
    result = r.run()
    print(f"Run complete: {result}")


def _doctor() -> None:
    from foundry.backends import detect_backend
    info = detect_backend()

    W = 54   # column width
    print()
    print("─" * W)
    print("  olaverse-foundry — environment check")
    print("─" * W)

    # Python
    print(f"  {'Python':20s}  {info['python_version']}")

    # Torch
    if info["torch"]:
        tv = info["torch_version"] or "?"
        if info["cuda"]:
            vram = info["gpu_vram_gb"]
            vram_str = (
                ", ".join(f"{v} GB" for v in vram)
                if vram else "?"
            )
            device_str = (
                f"CUDA {info['cuda_version']}, "
                f"{info['gpu_count']}× GPU [{vram_str}]"
            )
        elif info["mps"]:
            device_str = "MPS / Apple Silicon"
        else:
            device_str = "CPU only"
        print(f"  {'torch':20s}  ✓  {tv}  ({device_str})")
    else:
        print(f"  {'torch':20s}  ✗  NOT INSTALLED")

    # Optional packages
    optional = [
        ("accelerate",  "distributed training (CachedDistillTrainer)"),
        ("safetensors", "fast/safe weight format (PEFT adapter save)"),
        ("peft",        "LoRA skill packs (lego backend)"),
        ("mergekit",    "SOLAR depth upscale (merge backend)"),
        ("rapidfuzz",   "fast MinED alignment (align backend, optional)"),
        ("wandb",       "experiment tracking (set log_backend='wandb')"),
    ]
    for key, label in optional:
        mark = "✓" if info[key] else "✗"
        print(f"  {key:20s}  {mark}  {label}")

    print("─" * W)

    # Install hints for missing required/recommended packages
    missing = []
    if not info["torch"]:
        missing.append(("torch + training",  "pip install olaverse-foundry[torch]"))
    if not info["accelerate"]:
        missing.append(("accelerate",        "pip install olaverse-foundry[torch]"))
    if not info["safetensors"]:
        missing.append(("safetensors",       "pip install olaverse-foundry[torch]"))
    if not info["peft"]:
        missing.append(("peft (LoRA packs)", "pip install olaverse-foundry[lego]"))
    if not info["mergekit"]:
        missing.append(("mergekit (growth)", "pip install olaverse-foundry[merge]"))
    if not info["rapidfuzz"]:
        missing.append(("rapidfuzz (align)", "pip install olaverse-foundry[align]"))
    if not info["wandb"]:
        missing.append(("wandb (logging)",   "pip install olaverse-foundry[logging]"))

    if missing:
        print()
        print("  Optional installs:")
        for label, cmd in missing:
            print(f"    {label:22s}  {cmd}")

    print()
    if info["torch"] and (info["cuda"] or info["mps"]):
        print("  Status: GPU available — real training enabled.")
    elif info["torch"]:
        print("  Status: CPU-only — toy backend for testing; no real training.")
    else:
        print("  Status: numpy-only — toy backend only.")
    print()


def _strategies() -> None:
    from foundry.fusion.strategies import STRATEGY_REGISTRY
    print()
    print("  Fusion strategies (foundry.fusion.strategies)")
    print("  ─" * 27)
    descriptions = {
        "min_ce":  "MinCE   — per token, pick teacher with highest p(gold). [FuseLLM best]",
        "mean_ce": "MeanCE  — weighted average over all teacher distributions.",
    }
    for name in STRATEGY_REGISTRY:
        desc = descriptions.get(name, "custom — user-registered strategy")
        print(f"  {name:12s}  {desc}")
    print()
    print("  Register a custom strategy:")
    print("    from foundry.fusion import register_strategy")
    print("    @register_strategy('my_strategy')")
    print("    def my_fn(dists, gold_ids, weights): ...")
    print()


def _embed(recipe_path: str) -> None:
    """
    Run an embedding-distillation recipe end-to-end.

    Loads the student/teacher encoders, builds the training data from the
    recipe's ``data:`` block (HF dataset, streaming, raw text — handled by
    DataPipeline), and runs EmbeddingDistillTrainer to completion.
    """
    from foundry.recipes import Recipe
    from foundry.recipes.schema import EmbedRecipe

    recipe = Recipe.load(recipe_path)
    if not isinstance(recipe._spec, EmbedRecipe):
        print(
            "This recipe is not an embedding recipe.\n"
            "Add 'embed_loss' / 'embed_pool' under fusion:, or use `foundry run`."
        )
        sys.exit(1)

    if recipe._spec.data is None:
        print(
            "embed recipe needs a data: block so the CLI can load training data.\n"
            "Example:\n"
            "  data:\n"
            "    source: sentence-transformers/all-nli\n"
            "    split: train\n"
            "    text_column: anchor\n"
            "    batch_size: 32\n"
            "    max_length: 128"
        )
        sys.exit(1)

    result = recipe.run()   # builds DataPipeline from data: block, then trains
    print(f"[foundry embed] complete: {result}")


# ── Main ───────────────────────────────────────────────────────────────────

_COMMANDS = {
    "plan":       ("plan   <recipe.yaml>", "show staged plan and cost estimates"),
    "run":        ("run    <recipe.yaml>", "execute the recipe"),
    "doctor":     ("doctor",              "check installed backends"),
    "strategies": ("strategies",          "list available fusion strategies"),
    "embed":      ("embed  <recipe.yaml>", "run an embedding distillation recipe"),
}


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print()
        print("  olaverse-foundry CLI")
        print()
        print("  Usage: foundry <command> [args]")
        print()
        for usage, desc in _COMMANDS.values():
            print(f"    foundry {usage:30s}  {desc}")
        print()
        sys.exit(0)

    cmd, *rest = args

    if cmd == "plan":
        if not rest:
            print("Usage: foundry plan <recipe.yaml>")
            sys.exit(1)
        _plan(rest[0])

    elif cmd == "run":
        if not rest:
            print("Usage: foundry run <recipe.yaml>")
            sys.exit(1)
        _run(rest[0])

    elif cmd == "doctor":
        _doctor()

    elif cmd == "strategies":
        _strategies()

    elif cmd == "embed":
        if not rest:
            print("Usage: foundry embed <recipe.yaml>")
            sys.exit(1)
        _embed(rest[0])

    else:
        print(f"Unknown command: {cmd!r}")
        print("Run `foundry --help` for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
