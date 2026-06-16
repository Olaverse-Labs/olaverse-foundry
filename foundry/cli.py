"""
foundry CLI — plan / run / doctor / strategies

Usage:
    foundry plan   recipe.yaml
    foundry run    recipe.yaml
    foundry doctor
    foundry strategies
"""
from __future__ import annotations

import sys


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
    print("\n── foundry doctor ──────────────────────────────────────")
    checks = {
        "torch":      "PyTorch (required for real training)",
        "cuda":       "CUDA GPU",
        "mps":        "Apple MPS GPU",
        "peft":       "peft (LoRA skill packs)",
        "accelerate": "accelerate (distributed training)",
        "mergekit":   "mergekit (growth backend)",
    }
    for key, label in checks.items():
        status = "✓" if info[key] else "✗"
        print(f"  {status}  {label}")
    print()
    if not info["torch"]:
        print("  Install torch backend:  pip install olaverse-foundry[torch]")
    if not info["peft"]:
        print("  Install lego backend:   pip install olaverse-foundry[lego]")
    if not info["mergekit"]:
        print("  Install merge backend:  pip install olaverse-foundry[merge]")
    print()


def _strategies() -> None:
    from foundry.fusion.strategies import STRATEGY_REGISTRY
    print("\n── Fusion strategies ───────────────────────────────────")
    descriptions = {
        "min_ce": "MinCE — per token, pick teacher with highest p(gold). (FuseLLM best)",
        "mean":   "Mean  — weighted average over all teacher distributions.",
    }
    for name in STRATEGY_REGISTRY:
        desc = descriptions.get(name, "custom strategy")
        print(f"  {name:12s}  {desc}")
    print()


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("Usage: foundry <command> [args]\n")
        print("Commands:")
        print("  plan   <recipe.yaml>  — show staged plan and cost estimates (no compute)")
        print("  run    <recipe.yaml>  — execute the recipe")
        print("  doctor                — check installed backends")
        print("  strategies            — list available fusion strategies")
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

    else:
        print(f"Unknown command: {cmd!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
