"""Continuous, task-agnostic compatibility scoring for Skill Memory.

``SkillMemoryPlugin`` needs a ``compatibility(record, query) -> float`` score
in ``[0, 1]`` for every stored skill. Two ad-hoc scorers already existed in
the example benchmarks (an arithmetic "is this operation a prerequisite?"
lookup and a RotatedMNIST "does the rotation match exactly?" lookup). Both
are binary: they only ever return exactly ``0.0`` or ``1.0``. Because the
score is the trigger for REUSE/CLONE/SCRATCH, a binary scorer can never
produce a value inside the CLONE band (``clone_threshold <= score <
reuse_threshold``) — the decision degenerates to a two-way SCRATCH/REUSE
choice no matter how the thresholds are set. This is the "transition is not
automatic" problem: the CLONE regime is never exercised by construction, not
because CLONE is a bad idea for the task.

``ProbeCompatibilityScorer`` replaces both ad-hoc functions with a single,
principled, continuous score: load the candidate skill into a fresh copy of
the model architecture and measure its *zero-shot* error on a small probe
drawn from the *new* experience's own training data (never the held-out
evaluation stream, so there is no test leakage). The probe error is compared
against a cheap reference error (e.g. a mean/majority-class predictor) and
mapped smoothly into ``[0, 1]``:

* ``score -> 1``  the candidate already solves the new experience zero-shot
  (safe to REUSE outright).
* ``score -> 0``  the candidate is no better than the naive reference (start
  from SCRATCH).
* in between      the candidate captures some but not all of the new
  structure (CLONE: use it as initialization, then keep training).

This is deliberately generic: the same class works for the arithmetic PoC
and for RotatedMNIST by swapping ``loss_fn``/``probe_fn``/``reference_fn``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

import torch
from torch import Tensor, nn

from .memory import SkillRecord


@dataclass
class ProbeCompatibilityScorer:
    """Score a stored skill by its zero-shot error on a small live probe.

    Parameters
    ----------
    model_factory:
        Builds a fresh, untrained model with the same architecture used for
        training. Called once per scored candidate so probing never mutates
        the strategy's live model or the stored skill.
    loss_fn:
        ``(predictions, targets) -> scalar tensor``. Lower is better (e.g.
        ``nn.functional.mse_loss`` or ``nn.functional.cross_entropy``).
    probe_fn:
        ``query -> (x, y)``. Must draw only from the new experience's own
        training data (a small held-out-from-training-loss subset is fine;
        the evaluation/test stream must never be used here).
    reference_fn:
        ``query -> float``. A cheap, data-driven upper-bound error for the
        probe (e.g. the error of predicting the probe-target mean, or the
        entropy of a uniform-class guess). Used to normalize the candidate's
        probe error onto a roughly task-scale-independent ``[0, 1]`` range.
    min_reference:
        Reference values at or below this are treated as degenerate (a
        reference of ~0 would make any nonzero error score below 0), and the
        score is clamped to ``0.0`` rather than dividing by ~0.
    """

    model_factory: Callable[[], nn.Module]
    loss_fn: Callable[[Tensor, Tensor], Tensor]
    probe_fn: Callable[[Any], Tuple[Tensor, Tensor]]
    reference_fn: Callable[[Any], float]
    min_reference: float = 1e-8

    def __call__(self, record: SkillRecord, query: Any) -> float:
        model = self.model_factory()
        model.load_state_dict(record.state_dict)
        model.eval()

        x, y = self.probe_fn(query)
        with torch.no_grad():
            predictions = model(x)
            probe_error = float(self.loss_fn(predictions, y))

        reference_error = float(self.reference_fn(query))
        if reference_error <= self.min_reference:
            # Degenerate probe (e.g. a constant target): fall back to an
            # absolute comparison instead of a ratio.
            return 0.0 if probe_error > self.min_reference else 1.0

        score = 1.0 - (probe_error / reference_error)
        return max(0.0, min(1.0, score))


def mean_baseline_mse(y: Tensor) -> float:
    """Reference MSE of predicting the probe-target mean for every sample.

    A convenient ``reference_fn`` for regression probes: it is computed
    entirely from the probe's own targets, so it needs no extra data and
    adapts automatically to each task's output scale.
    """

    mean_prediction = y.mean(dim=0, keepdim=True)
    return float(nn.functional.mse_loss(mean_prediction.expand_as(y), y))


def uniform_guess_cross_entropy(num_classes: int) -> Callable[[Any], float]:
    """Reference cross-entropy of guessing uniformly over ``num_classes``.

    A convenient ``reference_fn`` for classification probes: guessing
    uniformly has a fixed, distribution-independent cross-entropy of
    ``log(num_classes)``, so this does not need to inspect the probe at all.
    """

    import math

    reference = math.log(num_classes)

    def _reference_fn(_query: Any) -> float:
        return reference

    return _reference_fn
