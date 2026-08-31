"""Reusable skill-memory primitives for continual-learning strategies."""

from .memory import SkillMemory, SkillMemoryError, SkillNotFoundError, SkillRecord
from .plugin import SkillMemoryPlugin
from .compatibility import (
    ProbeCompatibilityScorer,
    mean_baseline_mse,
    uniform_guess_cross_entropy,
)

__all__ = [
    "SkillMemory",
    "SkillMemoryError",
    "SkillNotFoundError",
    "SkillRecord",
    "SkillMemoryPlugin",
    "ProbeCompatibilityScorer",
    "mean_baseline_mse",
    "uniform_guess_cross_entropy",
]
