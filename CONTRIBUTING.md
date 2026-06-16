# Contributing to olaverse-foundry

Thank you for your interest in contributing! `olaverse-foundry` is the model-building layer of the [Olaverse ecosystem](https://olaverse.co.uk) — tools for distilling, growing, fusing, and packaging LLMs and embedding models.

---

## Ways to contribute

- **Bug reports** — open an [issue](https://github.com/Olaverse-Labs/olaverse-foundry/issues) with the bug report template
- **Feature requests** — open an issue with the feature request template
- **Pull requests** — fix a bug, add a feature, improve docs
- **Documentation** — improve examples, add recipes, fix typos
- **Testing** — add test cases for edge cases or untested trainers

---

## Development setup

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/olaverse-foundry.git
cd olaverse-foundry

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev extras
pip install -e ".[torch,lego,data,dev]"

# 4. Run the test suite
pytest tests/ -v
```

---

## Project structure

```
olaverse-foundry/
├── foundry/
│   ├── training/          # TorchDistillTrainer, CachedDistillTrainer, EmbeddingDistillTrainer
│   │   ├── torch_distill.py
│   │   ├── accelerate_distill.py
│   │   ├── embed_distill.py
│   │   ├── _logger.py     # W&B / TensorBoard wrapper
│   │   └── _scheduler.py  # LR scheduler factory
│   ├── data/              # DataPipeline
│   ├── teachers/          # TeacherRegistry, HFTeacher, LogitCache
│   ├── skillpacks/        # SkillPack, SkillRegistry, PEFT bridge
│   ├── fusion/            # FusionKernel, alignment, strategies
│   ├── growth/            # GrowthPlan, mergekit backend
│   ├── recipes/           # Pydantic recipe schemas
│   ├── io/                # Model loading, seed
│   ├── backends.py        # detect_backend()
│   └── cli.py             # foundry CLI
├── tests/
├── docs/
└── pyproject.toml
```

---

## Contribution guidelines

### Code style

- Python 3.9+ compatible
- No formatting tool enforced — match the style of the surrounding code
- Imports: standard library → third-party → foundry-internal
- No comments unless the WHY is non-obvious

### Adding a new trainer

1. Create `foundry/training/my_trainer.py`
2. Define a `@dataclass` config that inherits from `TrainConfig` or `TorchTrainConfig`
3. Implement `train()` returning `{"losses": [...], "eval_losses": {...}, "device": str}`
4. Export from `foundry/training/__init__.py` and `foundry/__init__.py`
5. Add tests in `tests/test_my_trainer.py`
6. Document in `docs/training/`

### Adding a new fusion strategy

1. Add the function to `foundry/fusion/strategies.py`
2. Register it in `STRATEGY_REGISTRY`
3. Add it to `foundry strategies` CLI output
4. Document in `docs/recipes.md`

### Tests

- Tests live in `tests/` and use `pytest`
- All tests must run without GPU (use `ToyTeacher`, `TinyLM` stubs)
- New production features need tests in `tests/test_prod.py`
- Run the full suite before submitting: `pytest tests/ -v`

### Commits

- Use imperative present tense: `add`, `fix`, `update`, `remove`
- Keep subject line under 72 characters
- Reference issue numbers when applicable: `fix: handle empty dataset (#42)`

### Pull requests

- One logical change per PR
- Fill in the PR template
- All existing tests must pass
- New features need new tests

---

## Reporting a bug

1. Check [existing issues](https://github.com/Olaverse-Labs/olaverse-foundry/issues) first
2. Open a new issue using the **Bug Report** template
3. Include: Python version, OS, `pip show olaverse-foundry`, minimal repro code

---

## Requesting a feature

Open an issue using the **Feature Request** template. Describe the use case — not just the API you want.

---

## Questions?

Open a [discussion](https://github.com/Olaverse-Labs/olaverse-foundry/discussions) or email **hello@olaverse.co.uk**.

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
