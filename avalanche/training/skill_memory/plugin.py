"""Avalanche plugin for score-driven skill acquisition."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Optional

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from .memory import SkillMemory, SkillRecord


class SkillMemoryPlugin(SupervisedPlugin):
    """Choose REUSE, CLONE, or SCRATCH for each sequential experience.

    REUSE activates an existing stored skill without training or registering a
    new skill. CLONE starts from an independent copy of a stored skill, resets
    the optimizer state, trains it, and registers the result as a new skill.
    SCRATCH restores the model's initial state, resets the optimizer, trains it,
    and registers the result as a new skill. The registry itself remains
    immutable.
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
        force_decision: Optional[str] = None,
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
        self.force_decision = force_decision

        self.last_decision: str = self.SCRATCH
        self.last_selected_skill: Optional[str] = None
        self.last_reused_skill: Optional[str] = None
        self.last_compatibility_score: float = 0.0
        self._saved_train_epochs: Optional[int] = None
        self._initial_model_state: Optional[dict[str, Any]] = None

    @staticmethod
    def _reset_optimizer(strategy) -> None:
        """Start a newly acquired skill with fresh optimizer state."""
        optimizer = getattr(strategy, "optimizer", None)
        if optimizer is not None:
            optimizer.state.clear()

    def _capture_initial_model_state(self, strategy) -> None:
        """Capture the model initialization used by the first experience."""
        if self._initial_model_state is None:
            self._initial_model_state = {
                key: value.detach().cpu().clone()
                for key, value in strategy.model.state_dict().items()
            }

    def _restore_initial_model_state(self, strategy) -> None:
        """Restore a fresh acquisition state for SCRATCH."""
        if self._initial_model_state is None:
            raise RuntimeError("initial model state has not been captured")
        strategy.model.load_state_dict(deepcopy(self._initial_model_state))

    def before_training_exp(self, strategy, **kwargs):
        """Select acquisition mode and initialize the active learner."""
        self._capture_initial_model_state(strategy)
        self.last_decision = self.SCRATCH
        self.last_selected_skill = None
        self.last_reused_skill = None
        self.last_compatibility_score = 0.0
        self._saved_train_epochs = None

        if self.compatibility is None or len(self.memory) == 0:
            self._restore_initial_model_state(strategy)
            self._reset_optimizer(strategy)
            return

        if self.force_decision == self.SCRATCH:
            _, self.last_compatibility_score = self.memory.best_match(
                strategy.experience, self.compatibility, threshold=0.0
            )
            self._restore_initial_model_state(strategy)
            self._reset_optimizer(strategy)
            return

        query_threshold = 0.0 if self.force_decision else self.clone_threshold
        record, score = self.memory.best_match(
            strategy.experience, self.compatibility, threshold=query_threshold
        )
        self.last_compatibility_score = score
        if record is None:
            self._restore_initial_model_state(strategy)
            self._reset_optimizer(strategy)
            return

        self.last_selected_skill = record.name
        decision = self.force_decision
        if decision is None:
            decision = self.REUSE if score >= self.reuse_threshold else self.CLONE
        self.last_decision = decision

        if decision == self.REUSE:
            self.last_reused_skill = record.name
            self.memory.load_into(record.name, strategy.model)
            if hasattr(strategy, "train_epochs"):
                self._saved_train_epochs = strategy.train_epochs
                strategy.train_epochs = 0
            return

        # CLONE starts from the stored source state, but the optimizer belongs
        # to the new acquisition and must not carry state from the source's
        # previous training history.
        if decision == self.CLONE:
            self.memory.load_into(record.name, strategy.model)
            self._reset_optimizer(strategy)
            return

        # A forced decision can only be one of the three constants. Keep the
        # fallback explicit in case this code is extended later.
        self._restore_initial_model_state(strategy)
        self._reset_optimizer(strategy)

    def after_training_exp(self, strategy, **kwargs):
        """Restore settings and register only newly acquired skills."""
        if self._saved_train_epochs is not None:
            strategy.train_epochs = self._saved_train_epochs
            self._saved_train_epochs = None

        # REUSE is a use operation, not an acquisition operation. The source
        # record remains unchanged and no new memory record is created.
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
