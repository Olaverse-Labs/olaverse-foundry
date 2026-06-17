"""
olaverse-foundry — a toolkit for building model families.

One expensive ancestor, many cheap descendants.

    seed → grow → fuse/heal → freeze → skill packs

Quick start::

    from foundry import Recipe

    recipe = Recipe.load("my_recipe.yaml")
    for line in recipe.plan():
        print(line)

    base = recipe.run()
"""
from foundry.contracts import ArchConfig, Student, Teacher, TokenizerAlignment
from foundry.fusion import FusionKernel, IdentityAlignment, EMAlignment, MinEDAlignment
from foundry.fusion import STRATEGY_REGISTRY, min_ce, mean_ce
from foundry.growth import (
    GrowthPlan, upscale_layer_map, layers_for_param_target, plan_growth,
    build_upscaled_state_dict,
    growth_plan_to_mergekit_yaml, save_mergekit_config, run_merge,
)
from foundry.skillpacks import (
    SkillPack, SkillRegistry,
    save_as_peft, load_from_peft, peft_config_dict,
)
from foundry.teachers import TeacherRegistry, ToyTeacher, HFTeacher, LogitCache
from foundry.training import (
    DistillTrainer, TrainConfig,
    TorchDistillTrainer, TorchTrainConfig,
    CachedDistillTrainer, CachedDistillConfig,
    EmbeddingDistillTrainer, EmbeddingDistillConfig, ToyEmbeddingTeacher,
)
from foundry.io import SeedResult, load_seed
from foundry.recipes import Recipe, FoundryRecipe, EmbedRecipe, EmbedFusionConfig, DataConfig
from foundry.backends import detect_backend
from foundry.data import DataPipeline

__version__ = "0.1.0"

__all__ = [
    # Contracts
    "ArchConfig", "Student", "Teacher", "TokenizerAlignment",
    # Fusion
    "FusionKernel", "IdentityAlignment", "EMAlignment", "MinEDAlignment",
    "STRATEGY_REGISTRY", "min_ce", "mean_ce",
    # Growth
    "GrowthPlan", "upscale_layer_map", "layers_for_param_target", "plan_growth",
    "build_upscaled_state_dict",
    "growth_plan_to_mergekit_yaml", "save_mergekit_config", "run_merge",
    # Skill packs
    "SkillPack", "SkillRegistry",
    "save_as_peft", "load_from_peft", "peft_config_dict",
    # Teachers
    "TeacherRegistry", "ToyTeacher", "HFTeacher", "LogitCache",
    # Training
    "DistillTrainer", "TrainConfig",
    "TorchDistillTrainer", "TorchTrainConfig",
    "CachedDistillTrainer", "CachedDistillConfig",
    "EmbeddingDistillTrainer", "EmbeddingDistillConfig", "ToyEmbeddingTeacher",
    # IO / Seed
    "SeedResult", "load_seed",
    # Recipes
    "Recipe", "FoundryRecipe", "EmbedRecipe", "EmbedFusionConfig", "DataConfig",
    # Backends
    "detect_backend",
    # Data
    "DataPipeline",
]
