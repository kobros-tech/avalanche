"""Avalanche plugin for score-driven skill acquisition."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from .memory import SkillMemory, SkillRecord


class SkillMemoryPlugin(SupervisedPlugin):
    """Choose REUSE, CLONE, or SCRATCH for each sequential experience.

    REUSE does not train the selected source skill. CLONE copies the selected
    source into a new acquisition state and then trains that independent copy.
    SCRATCH trains a fresh acquisition state. The registry itself remains
    immutable.

    This plugin provides the generic policy/lifecycle layer. Architectures that
    reserve explicit neuron/parameter slots can additionally use the slot
    adapter described in ``docs/skill_memory_algorithm.md``.
    """

    REUSE = "reuse"
    CLONE = "clone"
    SCRATCH = "scratch"

    def __init__(
        self,
        memory: Optional[SkillMemory] = None,
        *,
        skill_name: Callable[[Any], str] = lambda exp: str(exp.current_experience),
        compatibility: Optional[Callable[[SkillRecord, Any], float]] = None,
        skill_metadata: Optional[Callable[[Any], Mapping[str, Any]]] = None,
        reuse_threshold: float = 0.90,
        clone_threshold: float = 0.30,
        threshold: Optional[float] = None,
        replace_existing: bool = False,
        reset_optimizer_on_reuse: bool = False,
        force_decision: Optional[str] = None,
        train_on_reuse: bool = False,
    ):
        super().__init__()
        if force_decision is not None and force_decision not in (
            self.REUSE, self.CLONE, self.SCRATCH
        ):
            raise ValueError(
                "force_decision must be one of 'reuse', 'clone', 'scratch', or None"
            )
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
        self.skill_metadata = skill_metadata
        self.reuse_threshold = reuse_threshold
        self.clone_threshold = clone_threshold
        self.replace_existing = replace_existing
        self.reset_optimizer_on_reuse = reset_optimizer_on_reuse
        self.force_decision = force_decision
        self.train_on_reuse = train_on_reuse

        self.last_decision: str = self.SCRATCH
        self.last_selected_skill: Optional[str] = None
        self.last_reused_skill: Optional[str] = None
        self.last_compatibility_score: float = 0.0
        self._saved_train_epochs: Optional[int] = None

    def before_training_exp(self, strategy, **kwargs):
        """Select acquisition mode and initialize the active learner."""
        self.last_decision = self.SCRATCH
        self.last_selected_skill = None
        self.last_reused_skill = None
        self.last_compatibility_score = 0.0
        self._saved_train_epochs = None

        if self.compatibility is None or len(self.memory) == 0:
            return

        if self.force_decision == self.SCRATCH:
            _, self.last_compatibility_score = self.memory.best_match(
                strategy.experience, self.compatibility, threshold=0.0
            )
            return

        query_threshold = 0.0 if self.force_decision else self.clone_threshold
        record, score = self.memory.best_match(
            strategy.experience, self.compatibility, threshold=query_threshold
        )
        self.last_compatibility_score = score
        if record is None:
            return

        self.last_selected_skill = record.name
        decision = self.force_decision
        if decision is None:
            decision = self.REUSE if score >= self.reuse_threshold else self.CLONE
        self.last_decision = decision
        if decision == self.REUSE:
            self.last_reused_skill = record.name

        # load_into is an independent state copy. Training the active model
        # therefore cannot mutate the stored source record.
        self.memory.load_into(record.name, strategy.model)

        if decision == self.REUSE and not self.train_on_reuse:
            # Direct REUSE means use the learned skill as-is. Temporarily
            # suppress ordinary Avalanche training for this experience.
            if hasattr(strategy, "train_epochs"):
                self._saved_train_epochs = strategy.train_epochs
                strategy.train_epochs = 0
        elif decision == self.REUSE and self.reset_optimizer_on_reuse:
            if hasattr(strategy, "optimizer"):
                strategy.optimizer.state.clear()

    def after_training_exp(self, strategy, **kwargs):
        """Restore settings and register only newly acquired skills."""
        if self._saved_train_epochs is not None:
            strategy.train_epochs = self._saved_train_epochs
            self._saved_train_epochs = None

        # REUSE references an existing immutable skill; it does not acquire a
        # new version and must never replace the source record.
        if self.last_decision == self.REUSE:
            return

        experience = strategy.experience
        name = self.skill_name(experience)
        metadata = dict(self.skill_metadata(experience)) if self.skill_metadata else {}
        metadata.update(
            {
                "acquisition_decision": self.last_decision,
                "selected_skill": self.last_selected_skill,
                "reused_from": self.last_reused_skill,
                "compatibility_score": self.last_compatibility_score,
            }
        )

        if self.memory.contains(name) and not self.replace_existing:
            return

        self.memory.register(
            name,
            strategy.model.state_dict(),
            metadata=metadata,
            replace=self.replace_existing,
        )
