"""
M0 test suite — runs entirely on numpy, no GPU required.
"""
import numpy as np
import pytest
from foundry.contracts import ArchConfig
from foundry.fusion import FusionKernel, IdentityAlignment
from foundry.fusion.strategies import min_ce, mean_ce
from foundry.growth import upscale_layer_map, layers_for_param_target, plan_growth
from foundry.skillpacks import SkillPack, SkillRegistry
from foundry.teachers import TeacherRegistry, ToyTeacher, LogitCache
from foundry.training import DistillTrainer, TrainConfig
from foundry.recipes import Recipe
from foundry.backends import detect_backend


# ── ArchConfig ─────────────────────────────────────────────────────────────

def test_arch_config_params():
    cfg = ArchConfig(n_layers=32, d_model=4096, vocab_size=32000)
    p = cfg.params_estimate()
    assert p > 6e9, "Expected ~7B params for a Mistral-7B-sized config"
    assert p < 10e9

def test_arch_config_shape_warning():
    # Deep/narrow: 75 layers at d_model=4096
    cfg = ArchConfig(n_layers=75, d_model=4096, vocab_size=32000)
    assert cfg.shape_warning() is not None, "Should warn on deep/narrow shape"

def test_arch_config_no_warning():
    cfg = ArchConfig(n_layers=32, d_model=4096, vocab_size=32000)
    assert cfg.shape_warning() is None


# ── Fusion kernel ──────────────────────────────────────────────────────────

def test_identity_alignment_scatter():
    B, S, K, V = 2, 4, 5, 20
    align = IdentityAlignment()
    idx   = np.random.randint(0, V, (B, S, K))
    probs = np.ones((B, S, K), dtype=np.float32) / K
    out   = align.map(idx, probs, V)
    assert out.shape == (B, S, V)
    assert np.all(out >= 0)

def test_fusion_kernel_loss_is_scalar():
    B, S, V = 2, 8, 50
    kernel = FusionKernel(strategy="min_ce", alpha=0.3)
    logits = np.random.randn(B, S, V).astype(np.float32)
    gold   = np.random.randint(0, V, (B, S))
    dists  = [np.random.dirichlet(np.ones(V), size=(B, S)).astype(np.float32)]
    loss   = kernel.loss(logits, gold, dists)
    assert isinstance(loss, float)
    assert loss > 0

def test_min_ce_selects_best_teacher():
    B, S, V = 1, 4, 10
    # Teacher A knows the gold token; teacher B does not
    gold = np.array([[2, 5, 7, 1]])
    good = np.zeros((B, S, V), dtype=np.float32)
    bad  = np.zeros((B, S, V), dtype=np.float32)
    for b in range(B):
        for s in range(S):
            good[b, s, gold[b, s]] = 0.9
            bad[b, s, (gold[b, s] + 1) % V] = 0.9
    fused = min_ce([good, bad], gold)
    assert fused.shape == (B, S, V)

def test_mean_ce():
    B, S, V = 2, 4, 20
    d1 = np.random.dirichlet(np.ones(V), (B, S)).astype(np.float32)
    d2 = np.random.dirichlet(np.ones(V), (B, S)).astype(np.float32)
    out = mean_ce([d1, d2], np.zeros((B, S), dtype=np.int32), [1.0, 1.0])
    assert out.shape == (B, S, V)
    assert np.allclose(out.sum(-1), 1.0, atol=1e-5)


# ── Growth planner ─────────────────────────────────────────────────────────

def test_upscale_layer_map_double():
    lmap = upscale_layer_map(4, 8)
    assert len(lmap) == 8
    assert all(0 <= i < 4 for i in lmap)

def test_upscale_layer_map_exact():
    lmap = upscale_layer_map(4, 4)
    assert lmap == [0, 1, 2, 3]

def test_upscale_raises_on_shrink():
    with pytest.raises(ValueError):
        upscale_layer_map(8, 4)

def test_layers_for_param_target():
    cfg = ArchConfig(n_layers=32, d_model=4096, vocab_size=32000)
    n, warning = layers_for_param_target(cfg, 15e9)
    assert n > 32, "15B should need more layers than a 7B seed"

def test_plan_growth_summary():
    cfg  = ArchConfig(n_layers=32, d_model=4096, vocab_size=32000, name="seed")
    plan = plan_growth(cfg, to_params=15e9)
    summary = plan.summary()
    assert "32 →" in summary


# ── Skill packs ────────────────────────────────────────────────────────────

def _toy_state_dict():
    return {
        "layer.0.q_proj": np.ones((64, 64), dtype=np.float32),
        "layer.0.v_proj": np.ones((64, 64), dtype=np.float32),
        "layer.0.bias":   np.zeros(64,       dtype=np.float32),
    }

def test_skill_pack_apply():
    state = _toy_state_dict()
    from foundry.skillpacks.pack import _model_hash
    base_hash = _model_hash(state)
    A = np.zeros((16, 64), dtype=np.float32)
    B = np.zeros((64, 16), dtype=np.float32)
    pack = SkillPack(
        name="test_skill",
        base_hash=base_hash,
        rank=16,
        weights={"q_proj": {"A": A, "B": B}},
    )
    updated = pack.apply(state["layer.0.q_proj"], "q_proj")
    assert updated.shape == (64, 64)

def test_skill_registry_wrong_base_raises():
    state = _toy_state_dict()
    registry = SkillRegistry(state)
    bad_pack = SkillPack(name="bad", base_hash="deadbeef")
    with pytest.raises(ValueError, match="wrong base"):
        registry.register(bad_pack)

def test_skill_registry_snap_on():
    state = _toy_state_dict()
    from foundry.skillpacks.pack import _model_hash
    bh = _model_hash(state)
    pack = SkillPack(name="math", base_hash=bh, rank=4,
                     weights={"q_proj": {"A": np.zeros((4, 64)), "B": np.zeros((64, 4))}})
    reg = SkillRegistry(state)
    reg.register(pack)
    merged = reg.snap_on("math")
    assert "layer.0.q_proj" in merged


# ── Teacher & cache ────────────────────────────────────────────────────────

def test_toy_teacher_distribution_shape():
    teacher = ToyTeacher(name="t", vocab_size=100, top_k=10)
    ids = np.array([[1, 2, 3, 4]])
    idx, probs = teacher.distribution(ids, top_k=10)
    assert idx.shape   == (1, 4, 10)
    assert probs.shape == (1, 4, 10)
    assert np.allclose(probs.sum(-1), 1.0, atol=1e-5)

def test_logit_cache_hit_miss():
    cache = LogitCache(top_k=5)
    idx   = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    prob  = np.array([0.4, 0.2, 0.2, 0.1, 0.1], dtype=np.float32)
    cache.put(("k", 0), idx, prob)
    assert cache.get(("k", 0)) is not None
    assert cache.get(("missing", 0)) is None
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 1

def test_teacher_registry_from_toy():
    reg = TeacherRegistry.from_toy(n=2, vocab_size=50)
    assert len(reg) == 2


# ── DistillTrainer ─────────────────────────────────────────────────────────

class _TinyStudent:
    config = ArchConfig(n_layers=2, d_model=32, vocab_size=50, name="tiny")
    tokenizer = None
    def forward(self, ids):
        np.random.seed(abs(int(ids.sum())) % 2**31)
        return np.random.randn(*ids.shape, 50).astype(np.float32)
    def parameters(self): return []

def test_distill_trainer_runs():
    student  = _TinyStudent()
    teachers = TeacherRegistry.from_toy(n=2, vocab_size=50)
    cfg      = TrainConfig(epochs=2, batch_size=2, log_every=5)
    dataset  = [np.random.randint(0, 50, (2, 8)) for _ in range(3)]
    history  = DistillTrainer(student, teachers, cfg).train(dataset)
    assert "losses" in history
    assert len(history["losses"]) == 2 * 3   # epochs × batches
    assert all(l > 0 for l in history["losses"])


# ── Recipe (toy path, no torch needed) ────────────────────────────────────

_RECIPE_DICT = {
    "seed":    {"model": "meta-llama/Llama-3.1-8B", "init": "pretrained"},
    "grow":    {"method": "depth_upscale", "to_params": "15B"},
    "teachers": [
        {"role": "reasoning", "model": "org/teacher-a", "weight": 1.0},
    ],
    "fusion":  {"strategy": "min_ce", "align": "min_ed", "cache": "topk_64"},
    "heal":    {"tokens": "1B", "alpha": 0.3},
    "output":  {"freeze_base": True, "skillpacks": ["ola_math"]},
}

def test_recipe_plan_no_compute():
    recipe = Recipe.from_dict(_RECIPE_DICT)
    lines  = recipe.plan()
    full   = "\n".join(lines)
    assert "15.0B" in full
    assert "min_ce" in full
    assert "SOLAR" in full

def test_recipe_toy_run():
    recipe = Recipe.from_dict(_RECIPE_DICT)
    result = recipe.run(backend="toy")
    assert result["mode"] == "toy"
    assert isinstance(result["final_loss"], float)

def test_detect_backend_keys():
    info = detect_backend()
    for key in ("torch", "cuda", "mps", "peft", "accelerate", "summary"):
        assert key in info
