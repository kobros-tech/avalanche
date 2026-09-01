"""Avalanche plugin for evidence-driven skill acquisition."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Optional

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from .memory import SkillMemory, SkillRecord


class SkillMemoryPlugin(SupervisedPlugin):
    """Choose REUSE, CLONE, or SCRATCH for each sequential experience.

    When ``clone_compatibility`` is supplied, automatic policy selection uses
    zero-shot compatibility only for REUSE and uses matched post-adaptation
    value for CLONE. A clone is selected only when its measured improvement
    over a fresh model is positive; otherwise SCRATCH is the fallback.
    """

    REUSE = "reuse"
    CLONE = "clone"
    SCRATCH = "scratch"

    def __init__(self, memory: Optional[SkillMemory] = None, *,
                 skill_name: Callable[[Any], str] = lambda exp: str(exp.current_experience),
                 compatibility: Optional[Callable[[SkillRecord, Any], float]] = None,
                 clone_compatibility: Optional[Callable[[SkillRecord, Any], float]] = None,
                 skill_metadata: Optional[Callable[[Any], Mapping[str, Any]]] = None,
                 reuse_threshold: float = 0.90, clone_threshold: float = 0.30,
                 threshold: Optional[float] = None, replace_existing: bool = False,
                 force_decision: Optional[str] = None):
        super().__init__()
        if force_decision is not None and force_decision not in (self.REUSE, self.CLONE, self.SCRATCH):
            raise ValueError("force_decision must be one of 'reuse', 'clone', 'scratch', or None")
        if threshold is not None:
            clone_threshold = threshold
        if not 0.0 <= clone_threshold <= 1.0:
            raise ValueError("clone_threshold must be in [0, 1]")
        if not 0.0 <= reuse_threshold <= 1.0:
            raise ValueError("reuse_threshold must be in [0, 1]")
        if clone_threshold > reuse_threshold:
            raise ValueError("clone_threshold must not exceed reuse_threshold")

        self.memory = memory if memory is not None else SkillMemory()
        self.skill_name = skill_name
        self.compatibility = compatibility
        self.clone_compatibility = clone_compatibility
        self.skill_metadata = skill_metadata
        self.reuse_threshold = reuse_threshold
        self.clone_threshold = clone_threshold
        self.replace_existing = replace_existing
        self.force_decision = force_decision

        self.last_decision = self.SCRATCH
        self.last_selected_skill: Optional[str] = None
        self.last_reused_skill: Optional[str] = None
        self.last_compatibility_score = 0.0
        self.last_clone_value = 0.0
        self._saved_train_epochs: Optional[int] = None
        self._initial_model_state: Optional[dict[str, Any]] = None

    @staticmethod
    def _reset_optimizer(strategy) -> None:
        optimizer = getattr(strategy, "optimizer", None)
        if optimizer is not None:
            optimizer.state.clear()

    def _capture_initial_model_state(self, strategy) -> None:
        if self._initial_model_state is None:
            self._initial_model_state = {
                key: value.detach().cpu().clone()
                for key, value in strategy.model.state_dict().items()
            }

    def _restore_initial_model_state(self, strategy) -> None:
        if self._initial_model_state is None:
            raise RuntimeError("initial model state has not been captured")
        strategy.model.load_state_dict(deepcopy(self._initial_model_state))

    def _scratch(self, strategy) -> None:
        self._restore_initial_model_state(strategy)
        self._reset_optimizer(strategy)

    def before_training_exp(self, strategy, **kwargs):
        self._capture_initial_model_state(strategy)
        self.last_decision = self.SCRATCH
        self.last_selected_skill = None
        self.last_reused_skill = None
        self.last_compatibility_score = 0.0
        self.last_clone_value = 0.0
        self._saved_train_epochs = None

        if len(self.memory) == 0:
            self._scratch(strategy)
            return

        reuse_record = None
        reuse_score = 0.0
        if self.compatibility is not None:
            reuse_record, reuse_score = self.memory.best_match(
                strategy.experience, self.compatibility, threshold=0.0
            )
        self.last_compatibility_score = reuse_score

        if self.force_decision == self.SCRATCH:
            self._scratch(strategy)
            return

        if self.force_decision == self.REUSE:
            if reuse_record is None:
                self._scratch(strategy)
                return
            self.last_selected_skill = reuse_record.name
            self.last_reused_skill = reuse_record.name
            self.last_decision = self.REUSE
            self.memory.load_into(reuse_record.name, strategy.model)
            if hasattr(strategy, "train_epochs"):
                self._saved_train_epochs = strategy.train_epochs
                strategy.train_epochs = 0
            return

        if self.force_decision == self.CLONE:
            scorer = self.clone_compatibility or self.compatibility
            if scorer is None:
                self._scratch(strategy)
                return
            record, value = self.memory.best_match(
                strategy.experience, scorer, threshold=0.0
            )
            self.last_clone_value = value
            # Forced CLONE is an oracle/control condition: if a candidate exists,
            # clone it regardless of its score. The automatic policy below is
            # the place where positive evidence is required.
            if record is None:
                self._scratch(strategy)
                return
            self.last_selected_skill = record.name
            self.last_decision = self.CLONE
            self.memory.load_into(record.name, strategy.model)
            self._reset_optimizer(strategy)
            return

        if reuse_record is not None and reuse_score >= self.reuse_threshold:
            self.last_selected_skill = reuse_record.name
            self.last_reused_skill = reuse_record.name
            self.last_decision = self.REUSE
            self.memory.load_into(reuse_record.name, strategy.model)
            if hasattr(strategy, "train_epochs"):
                self._saved_train_epochs = strategy.train_epochs
                strategy.train_epochs = 0
            return

        if self.clone_compatibility is not None:
            record, value = self.memory.best_match(
                strategy.experience, self.clone_compatibility, threshold=0.0
            )
            self.last_clone_value = value
            if record is not None and value > 0.0:
                self.last_selected_skill = record.name
                self.last_decision = self.CLONE
                self.memory.load_into(record.name, strategy.model)
                self._reset_optimizer(strategy)
                return
        elif reuse_record is not None and reuse_score >= self.clone_threshold:
            self.last_selected_skill = reuse_record.name
            self.last_decision = self.CLONE
            self.memory.load_into(reuse_record.name, strategy.model)
            self._reset_optimizer(strategy)
            return

        self._scratch(strategy)

    def after_training_exp(self, strategy, **kwargs):
        if self._saved_train_epochs is not None:
            strategy.train_epochs = self._saved_train_epochs
            self._saved_train_epochs = None

        if self.last_decision == self.REUSE:
            return

        experience = strategy.experience
        name = self.skill_name(experience)
        metadata = dict(self.skill_metadata(experience)) if self.skill_metadata else {}
        metadata.update({
            "acquisition_decision": self.last_decision,
            "selected_skill": self.last_selected_skill,
            "reused_from": self.last_reused_skill,
            "compatibility_score": self.last_compatibility_score,
            "clone_value": self.last_clone_value,
        })

        if self.memory.contains(name) and not self.replace_existing:
            return

        self.memory.register(name, strategy.model.state_dict(), metadata=metadata,
                              replace=self.replace_existing)
