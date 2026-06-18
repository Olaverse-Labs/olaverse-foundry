"""
M4 tests — PEFT adapter bridge + mergekit config generation.

No mergekit, no PEFT library required — only safetensors (already installed)
and torch (already installed).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from foundry.skillpacks.pack import SkillPack, _model_hash
from foundry.skillpacks.peft_bridge import (
    peft_config_dict,
    save_as_peft,
    load_from_peft,
)
from foundry.growth.planner import plan_growth, GrowthPlan, upscale_layer_map
from foundry.growth.mergekit_backend import (
    _layer_map_to_slices,
    growth_plan_to_mergekit_yaml,
    save_mergekit_config,
    run_merge,
)
from foundry.contracts import ArchConfig


# ── Fixtures ───────────────────────────────────────────────────────────────

def _toy_pack(rank: int = 4) -> SkillPack:
    """Create a minimal SkillPack with random numpy weights."""
    d_in, d_out = 16, 32
    return SkillPack(
        name="test_skill",
        base_hash="abc123",
        rank=rank,
        alpha=float(rank),
        target_modules=["q_proj", "v_proj"],
        weights={
            "q_proj": {
                "A": np.random.randn(rank, d_in).astype(np.float32),
                "B": np.random.randn(d_out, rank).astype(np.float32),
            },
            "v_proj": {
                "A": np.random.randn(rank, d_in).astype(np.float32),
                "B": np.random.randn(d_out, rank).astype(np.float32),
            },
        },
    )


# ── peft_config_dict ────────────────────────────────────────────────────────

class TestPeftConfigDict(unittest.TestCase):

    def test_required_peft_fields(self):
        pack = _toy_pack()
        cfg  = peft_config_dict(pack)
        for field in ("peft_type", "task_type", "r", "lora_alpha", "target_modules"):
            self.assertIn(field, cfg, f"Missing: {field}")

    def test_rank_and_alpha_match_pack(self):
        pack = _toy_pack(rank=8)
        cfg  = peft_config_dict(pack)
        self.assertEqual(cfg["r"], 8)
        self.assertEqual(cfg["lora_alpha"], 8.0)

    def test_foundry_metadata_included(self):
        pack = _toy_pack()
        cfg  = peft_config_dict(pack)
        self.assertEqual(cfg["foundry_base_hash"], "abc123")
        self.assertEqual(cfg["foundry_name"],      "test_skill")

    def test_target_modules_preserved(self):
        pack = _toy_pack()
        cfg  = peft_config_dict(pack)
        self.assertEqual(set(cfg["target_modules"]), {"q_proj", "v_proj"})


# ── save_as_peft ────────────────────────────────────────────────────────────

class TestSaveAsPeft(unittest.TestCase):

    def test_creates_adapter_config(self):
        pack = _toy_pack()
        with tempfile.TemporaryDirectory() as tmp:
            save_as_peft(pack, tmp)
            self.assertTrue((Path(tmp) / "adapter_config.json").exists())

    def test_adapter_config_is_valid_json(self):
        pack = _toy_pack()
        with tempfile.TemporaryDirectory() as tmp:
            save_as_peft(pack, tmp)
            data = json.loads((Path(tmp) / "adapter_config.json").read_text())
            self.assertIn("r", data)

    def test_creates_weights_file(self):
        pack = _toy_pack()
        with tempfile.TemporaryDirectory() as tmp:
            save_as_peft(pack, tmp)
            st = (Path(tmp) / "adapter_model.safetensors").exists()
            pt = (Path(tmp) / "adapter_model.bin").exists()
            self.assertTrue(st or pt, "No weight file written")

    def test_returns_path(self):
        pack = _toy_pack()
        with tempfile.TemporaryDirectory() as tmp:
            result = save_as_peft(pack, tmp)
            self.assertEqual(Path(result), Path(tmp))

    def test_creates_output_dir_if_missing(self):
        pack = _toy_pack()
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = Path(tmp) / "nested" / "adapter"
            save_as_peft(pack, new_dir)
            self.assertTrue(new_dir.exists())


# ── load_from_peft ──────────────────────────────────────────────────────────

class TestLoadFromPeft(unittest.TestCase):

    def _save_and_load(self, pack: SkillPack, name_override=None) -> SkillPack:
        with tempfile.TemporaryDirectory() as tmp:
            save_as_peft(pack, tmp)
            return load_from_peft(tmp, name=name_override)

    def test_round_trip_rank(self):
        pack = _toy_pack(rank=8)
        restored = self._save_and_load(pack)
        self.assertEqual(restored.rank, 8)

    def test_round_trip_alpha(self):
        pack = _toy_pack(rank=4)
        restored = self._save_and_load(pack)
        self.assertAlmostEqual(restored.alpha, 4.0)

    def test_round_trip_name(self):
        pack = _toy_pack()
        restored = self._save_and_load(pack)
        self.assertEqual(restored.name, "test_skill")

    def test_round_trip_base_hash(self):
        pack = _toy_pack()
        restored = self._save_and_load(pack)
        self.assertEqual(restored.base_hash, "abc123")

    def test_round_trip_target_modules(self):
        pack = _toy_pack()
        restored = self._save_and_load(pack)
        self.assertEqual(set(restored.target_modules), {"q_proj", "v_proj"})

    def test_round_trip_weights_present(self):
        pack = _toy_pack()
        restored = self._save_and_load(pack)
        self.assertIn("q_proj", restored.weights)
        self.assertIn("v_proj", restored.weights)
        self.assertIn("A", restored.weights["q_proj"])
        self.assertIn("B", restored.weights["q_proj"])

    def test_round_trip_weight_values(self):
        pack = _toy_pack(rank=4)
        with tempfile.TemporaryDirectory() as tmp:
            save_as_peft(pack, tmp)
            restored = load_from_peft(tmp)
        np.testing.assert_allclose(
            restored.weights["q_proj"]["A"],
            pack.weights["q_proj"]["A"],
            rtol=1e-5,
        )

    def test_name_override(self):
        pack = _toy_pack()
        restored = self._save_and_load(pack, name_override="custom_name")
        self.assertEqual(restored.name, "custom_name")

    def test_restored_pack_apply_works(self):
        """Restored pack should apply LoRA delta without errors."""
        pack = _toy_pack(rank=4)
        with tempfile.TemporaryDirectory() as tmp:
            save_as_peft(pack, tmp)
            restored = load_from_peft(tmp)
        W = np.random.randn(32, 16).astype(np.float32)
        W_prime = restored.apply(W, "q_proj")
        self.assertEqual(W_prime.shape, W.shape)
        self.assertFalse(np.allclose(W_prime, W))  # delta was applied


# ── _layer_map_to_slices ────────────────────────────────────────────────────

class TestLayerMapToSlices(unittest.TestCase):

    def test_simple_double(self):
        """[0,1,2,3, 0,1,2,3] → 2 slices each [0,4)"""
        slices = _layer_map_to_slices([0, 1, 2, 3, 0, 1, 2, 3], "seed")
        self.assertEqual(len(slices), 2)
        self.assertEqual(slices[0]["sources"][0]["layer_range"], [0, 4])
        self.assertEqual(slices[1]["sources"][0]["layer_range"], [0, 4])

    def test_exact_copy(self):
        """[0,1,2,3] → 1 slice [0,4)"""
        slices = _layer_map_to_slices([0, 1, 2, 3], "seed")
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0]["sources"][0]["layer_range"], [0, 4])

    def test_interleaved(self):
        """[0,0,1,2,3,3] → 3 slices"""
        slices = _layer_map_to_slices([0, 0, 1, 2, 3, 3], "seed")
        self.assertEqual(len(slices), 3)
        self.assertEqual(slices[0]["sources"][0]["layer_range"], [0, 1])
        self.assertEqual(slices[1]["sources"][0]["layer_range"], [0, 4])
        self.assertEqual(slices[2]["sources"][0]["layer_range"], [3, 4])

    def test_seed_path_in_sources(self):
        slices = _layer_map_to_slices([0, 1], "my/model")
        self.assertEqual(slices[0]["sources"][0]["model"], "my/model")

    def test_single_layer(self):
        slices = _layer_map_to_slices([0], "seed")
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0]["sources"][0]["layer_range"], [0, 1])


# ── growth_plan_to_mergekit_yaml ───────────────────────────────────────────

class TestGrowthPlanToMergekitYaml(unittest.TestCase):

    def _plan(self) -> GrowthPlan:
        # 4-layer seed; target 2× → 8 layers
        cfg = ArchConfig(n_layers=4, d_model=256, vocab_size=1000)
        return plan_growth(cfg, to_params=5e7)

    def test_merge_method_is_passthrough(self):
        plan = self._plan()
        cfg  = growth_plan_to_mergekit_yaml(plan, "seed")
        self.assertEqual(cfg["merge_method"], "passthrough")

    def test_slices_present(self):
        plan = self._plan()
        cfg  = growth_plan_to_mergekit_yaml(plan, "seed")
        self.assertIn("slices", cfg)
        self.assertGreater(len(cfg["slices"]), 0)

    def test_dtype_field(self):
        plan = self._plan()
        cfg  = growth_plan_to_mergekit_yaml(plan, "seed", dtype="float16")
        self.assertEqual(cfg["dtype"], "float16")

    def test_foundry_note_included(self):
        plan = self._plan()
        cfg  = growth_plan_to_mergekit_yaml(plan, "seed")
        self.assertIn("_foundry_note", cfg)

    def test_sources_contain_seed_path(self):
        plan = self._plan()
        cfg  = growth_plan_to_mergekit_yaml(plan, "org/my-seed")
        for sl in cfg["slices"]:
            for src in sl["sources"]:
                self.assertEqual(src["model"], "org/my-seed")


# ── save_mergekit_config ───────────────────────────────────────────────────

class TestSaveMergekitConfig(unittest.TestCase):

    def test_creates_yaml_file(self):
        cfg  = ArchConfig(n_layers=4, d_model=256, vocab_size=1000)
        plan = plan_growth(cfg, to_params=5e7)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_mergekit_config(plan, "seed", Path(tmp) / "config.yaml")
            self.assertTrue(path.exists())

    def test_yaml_is_valid(self):
        import yaml
        cfg  = ArchConfig(n_layers=4, d_model=256, vocab_size=1000)
        plan = plan_growth(cfg, to_params=5e7)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_mergekit_config(plan, "seed", Path(tmp) / "config.yaml")
            data = yaml.safe_load(path.read_text())
            self.assertIn("merge_method", data)
            self.assertIn("slices",       data)

    def test_returns_path(self):
        cfg  = ArchConfig(n_layers=4, d_model=256, vocab_size=1000)
        plan = plan_growth(cfg, to_params=5e7)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub" / "merge.yaml"
            result = save_mergekit_config(plan, "seed", out)
            self.assertEqual(result, out)


# ── run_merge raises without mergekit ─────────────────────────────────────

class TestRunMergeNeedsTransformers(unittest.TestCase):

    def test_raises_import_error_without_transformers(self):
        # The native merge needs torch + transformers; raise clearly when absent.
        try:
            import transformers  # noqa: F401
            self.skipTest("transformers installed — native merge would run")
        except ImportError:
            pass
        import tempfile
        cfg  = ArchConfig(n_layers=4, d_model=256, vocab_size=1000)
        plan = plan_growth(cfg, to_params=5e7)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ImportError):
                run_merge(plan, "seed", tmp)


# ── Top-level foundry exports ──────────────────────────────────────────────

class TestFoundryM4Exports(unittest.TestCase):

    def test_m4_symbols_exported(self):
        import foundry
        for sym in (
            "save_as_peft", "load_from_peft", "peft_config_dict",
            "growth_plan_to_mergekit_yaml", "save_mergekit_config", "run_merge",
        ):
            self.assertIn(sym, foundry.__all__, f"Missing from __all__: {sym}")
            self.assertTrue(hasattr(foundry, sym), f"Not importable: {sym}")


if __name__ == "__main__":
    unittest.main()
