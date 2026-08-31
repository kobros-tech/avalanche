"""Standard Avalanche benchmark for Skill Memory: RotatedMNIST.

This experiment moves beyond the synthetic arithmetic proof of concept and
uses Avalanche's standard RotatedMNIST benchmark. The sequence intentionally
repeats transformations (0, 30, 60, 0, 30 degrees), creating legitimate
opportunities for reuse without hard-coding source/target task pairs.

Compatibility is scored with ``ProbeCompatibilityScorer``: each candidate
skill is loaded into a fresh model and evaluated on a small probe of the new
experience's own *training* images (never the held-out test stream), and its
cross-entropy is compared against the fixed reference of guessing uniformly
over the 10 digit classes (``log(10)``).

This replaces an earlier version of this benchmark whose compatibility
function returned exactly 1.0 for an identical rotation and exactly 0.0
otherwise. That binary scorer could only ever land in the SCRATCH/REUSE
extremes, so the CLONE band was structurally unreachable regardless of the
configured thresholds -- it wasn't that CLONE was tried and found unhelpful
for MNIST, it was never actually exercised. See "Why probe-based
compatibility" in docs/skill_memory_benchmark.md.

Conditions:
* naive: ordinary Avalanche Naive strategy.
* replay: Naive + Avalanche ReplayPlugin.
* skill_memory: Naive + SkillMemoryPlugin (automatic REUSE/CLONE/SCRATCH).
* skill_memory_scratch_only / skill_memory_clone_only / skill_memory_reuse_only:
  the same plugin with ``force_decision`` fixed, so the automatic policy can
  be compared against what each single strategy would have done on its own
  (the oracle-style comparison from Issue #3). These forced conditions only
  run for a reduced set of seeds/experiences (see ``ORACLE_SEEDS``) to keep
  the added cost small; they are diagnostic, not the headline comparison.

The experiment reports per-experience test accuracy, forgetting on previously
seen experiences, reuse/acquisition decisions, compatibility scores, and
training time.

Cost control: this benchmark trains on real MNIST images and was previously
removed from CI for being too expensive. Set the ``FAST_TEST=True``
environment variable (already used elsewhere in this repo's CI) to subsample
each experience's train/test data to ``FAST_TRAIN_SAMPLES`` /
``FAST_EVAL_SAMPLES`` images and run a single seed; this keeps the benchmark
fast enough for CI while still exercising the real code path end to end.
Without ``FAST_TEST`` it runs the full dataset across ``SEEDS`` for a
research-quality result.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from avalanche.benchmarks.classic import RotatedMNIST
from avalanche.models import SimpleMLP
from avalanche.training import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.skill_memory import (
    ProbeCompatibilityScorer,
    SkillMemory,
    SkillMemoryPlugin,
    uniform_guess_cross_entropy,
)


FAST_TEST = os.environ.get("FAST_TEST", "").lower() == "true"

SEEDS = [0] if FAST_TEST else [0, 1, 2]
# A small, separate slice of seeds that also run the forced-decision oracle
# conditions. Kept short because it triples the skill-memory cost.
ORACLE_SEEDS = [0] if FAST_TEST else [0, 1]

ROTATIONS = [0, 30, 60, 0, 30]
NUM_CLASSES = 10
EPOCHS = 1
TRAIN_MB_SIZE = 128
EVAL_MB_SIZE = 256
REPLAY_MEM_SIZE = 200
MEMORY_MAX_SKILLS = 5
REUSE_THRESHOLD = 0.90
CLONE_THRESHOLD = 0.30
PROBE_SAMPLES = 64

# Real MNIST has 60k/10k train/test images spread across 5 experiences; that
# is what made this benchmark expensive in CI. FAST_TEST subsamples each
# experience down to a small fixed size so the same code path still runs
# against real data, just less of it.
FAST_TRAIN_SAMPLES = 200
FAST_EVAL_SAMPLES = 200


def make_model() -> nn.Module:
    return SimpleMLP(num_classes=NUM_CLASSES)


class _SubsampledExperience:
    """Wraps an Avalanche experience, replacing ``dataset`` with a subset.

    Everything else (``current_experience``, ``origin_stream``, task labels,
    etc.) is forwarded to the original experience unchanged, so this is a
    drop-in substitute anywhere an experience is expected.
    """

    def __init__(self, experience, dataset):
        self.dataset = dataset
        self._experience = experience

    def __getattr__(self, name):
        return getattr(self._experience, name)


def subsample(experience, n_samples: int, seed: int) -> "_SubsampledExperience":
    """Return a copy of ``experience`` limited to ``n_samples`` items."""

    total = len(experience.dataset)
    n_samples = min(n_samples, total)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:n_samples].tolist()
    return _SubsampledExperience(experience, experience.dataset.subset(indices))


def maybe_subsample(experience, n_samples: int, seed: int):
    if not FAST_TEST:
        return experience
    return subsample(experience, n_samples, seed)


def accuracy(model: nn.Module, experience) -> float:
    """Evaluate one standard Avalanche test experience."""
    loader = DataLoader(experience.dataset, batch_size=EVAL_MB_SIZE, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, *rest in loader:
            logits = model(x)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == y).sum())
            total += int(y.numel())
    return correct / total if total else 0.0


def probe_batch(experience, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """A small probe drawn from the new experience's own *training* images.

    Independent from the actual training minibatches (a fresh random subset,
    seeded separately) and, crucially, independent from the evaluation/test
    stream, which this function never touches.
    """
    probed = subsample(experience, PROBE_SAMPLES, seed=seed + 900_000)
    loader = DataLoader(probed.dataset, batch_size=len(probed.dataset), shuffle=False)
    x, y, *_ = next(iter(loader))
    return x, y


def build_strategy(condition: str, seed: int):
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    plugins = []
    memory = None
    skill_plugin = None

    if condition == "replay":
        plugins.append(ReplayPlugin(mem_size=REPLAY_MEM_SIZE))
    elif condition.startswith("skill_memory"):
        memory = SkillMemory(max_skills=MEMORY_MAX_SKILLS)

        compatibility = ProbeCompatibilityScorer(
            model_factory=make_model,
            loss_fn=nn.functional.cross_entropy,
            probe_fn=lambda exp: probe_batch(exp, seed),
            reference_fn=uniform_guess_cross_entropy(NUM_CLASSES),
        )

        force_decision = {
            "skill_memory_scratch_only": SkillMemoryPlugin.SCRATCH,
            "skill_memory_clone_only": SkillMemoryPlugin.CLONE,
            "skill_memory_reuse_only": SkillMemoryPlugin.REUSE,
        }.get(condition)

        skill_plugin = SkillMemoryPlugin(
            memory=memory,
            skill_name=lambda exp: (
                f"rotation_{ROTATIONS[exp.current_experience]}_exp_{exp.current_experience}"
            ),
            skill_metadata=lambda exp: {"rotation": ROTATIONS[exp.current_experience]},
            compatibility=compatibility,
            reuse_threshold=REUSE_THRESHOLD,
            clone_threshold=CLONE_THRESHOLD,
            force_decision=force_decision,
        )
        plugins.append(skill_plugin)

    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        train_mb_size=TRAIN_MB_SIZE,
        train_epochs=EPOCHS,
        eval_mb_size=EVAL_MB_SIZE,
        eval_every=-1,
        plugins=plugins,
    )
    return strategy, memory, skill_plugin


def run_condition(condition: str, seed: int) -> Dict:
    torch.manual_seed(seed)
    benchmark = RotatedMNIST(
        n_experiences=len(ROTATIONS),
        rotations_list=ROTATIONS,
        seed=seed,
    )
    strategy, memory, plugin = build_strategy(condition, seed)

    accuracy_history: List[Dict[str, float]] = []
    decisions: List[Dict[str, object]] = []
    train_seconds = 0.0

    for raw_experience in benchmark.train_stream:
        exp_id = raw_experience.current_experience
        experience = maybe_subsample(
            raw_experience, FAST_TRAIN_SAMPLES, seed=seed * 100 + exp_id
        )

        started = time.perf_counter()
        strategy.train(experience, eval_streams=[])
        train_seconds += time.perf_counter() - started

        if condition == "naive":
            decision, score = "baseline", None
        elif condition == "replay":
            decision, score = "replay", None
        else:
            decision = plugin.last_decision
            if plugin.last_selected_skill is not None:
                decision = f"{decision}:{plugin.last_selected_skill}"
            score = plugin.last_compatibility_score

        decisions.append(
            {
                "experience": exp_id,
                "rotation": ROTATIONS[exp_id],
                "decision": decision,
                "compatibility_score": score,
            }
        )

        current = {}
        for test_index, raw_test_experience in enumerate(benchmark.test_stream):
            if test_index > exp_id:
                break
            test_experience = maybe_subsample(
                raw_test_experience, FAST_EVAL_SAMPLES, seed=seed * 100 + test_index
            )
            current[str(test_experience.current_experience)] = accuracy(
                strategy.model, test_experience
            )
        accuracy_history.append(current)

    forgetting = {}
    for exp_id in range(len(ROTATIONS) - 1):
        values = [row[str(exp_id)] for row in accuracy_history if str(exp_id) in row]
        forgetting[str(exp_id)] = max(0.0, max(values) - values[-1])

    return {
        "condition": condition,
        "seed": seed,
        "rotations": ROTATIONS,
        "epochs_per_experience": EPOCHS,
        "train_mb_size": TRAIN_MB_SIZE,
        "replay_mem_size": REPLAY_MEM_SIZE if condition == "replay" else None,
        "fast_test": FAST_TEST,
        "accuracy_history": accuracy_history,
        "final_accuracy": accuracy_history[-1],
        "forgetting_accuracy": forgetting,
        "decisions": decisions,
        "stored_skills": memory.names() if memory is not None else [],
        "train_seconds": train_seconds,
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def summarize(runs: List[Dict]) -> Dict:
    tasks = [str(i) for i in range(len(ROTATIONS))]
    final = {}
    for task in tasks:
        values = [run["final_accuracy"][task] for run in runs]
        final[task] = {"mean": mean(values), "std": std(values)}

    forgetting = {}
    for task in tasks[:-1]:
        values = [run["forgetting_accuracy"][task] for run in runs]
        forgetting[task] = {"mean": mean(values), "std": std(values)}

    return {
        "final_accuracy": final,
        "forgetting_accuracy": forgetting,
        "train_seconds": {
            "mean": mean(run["train_seconds"] for run in runs),
            "std": std(run["train_seconds"] for run in runs),
        },
    }


def main() -> None:
    all_results = []
    core_conditions = ("naive", "replay", "skill_memory")
    oracle_conditions = (
        "skill_memory_scratch_only",
        "skill_memory_clone_only",
        "skill_memory_reuse_only",
    )

    for seed in SEEDS:
        print(f"Running standard benchmark seed {seed}...")
        for condition in core_conditions:
            print(f"  condition={condition}")
            all_results.append(run_condition(condition, seed))

    for seed in ORACLE_SEEDS:
        print(f"Running oracle comparison seed {seed}...")
        for condition in oracle_conditions:
            print(f"  condition={condition}")
            all_results.append(run_condition(condition, seed))

    conditions = {}
    for condition in core_conditions + oracle_conditions:
        runs = [r for r in all_results if r["condition"] == condition]
        if not runs:
            continue
        conditions[condition] = {"runs": runs, "summary": summarize(runs)}

    result = {
        "benchmark": "skill_memory_rotated_mnist_v2",
        "description": (
            "Standard Avalanche RotatedMNIST benchmark comparing Naive, Replay, "
            "and Skill Memory (automatic, plus forced-decision oracle "
            "conditions), using ProbeCompatibilityScorer for continuous "
            "compatibility instead of a binary exact-rotation-match lookup."
        ),
        "fast_test": FAST_TEST,
        "seeds": SEEDS,
        "oracle_seeds": ORACLE_SEEDS,
        "rotations": ROTATIONS,
        "policy": {
            "reuse_threshold": REUSE_THRESHOLD,
            "clone_threshold": CLONE_THRESHOLD,
            "compatibility": "ProbeCompatibilityScorer (cross-entropy vs uniform-guess reference)",
        },
        "conditions": conditions,
    }

    output_path = Path("results/skill_memory_rotated_mnist.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for run in conditions["skill_memory"]["runs"]:
        decisions = [item["decision"] for item in run["decisions"]]
        # The first three experiences (rotations 0, 30, 60) are all new, so
        # memory is either empty or holds only unrelated rotations -- SCRATCH
        # is the only reachable decision there. Experiences 3 and 4 repeat an
        # earlier rotation exactly, so their best candidate's probe score
        # should be very high; whether that is enough to clear REUSE_THRESHOLD
        # (vs. landing in CLONE) is itself part of what this benchmark reports,
        # so it is not hard-asserted here -- see the written JSON results.
        assert decisions[0] == "scratch"
        assert decisions[1] == "scratch"
        assert decisions[2] == "scratch"
        for decision in decisions:
            assert (
                decision == "scratch"
                or decision.startswith("clone:")
                or decision.startswith("reuse:")
            )
        assert run["stored_skills"] == [
            f"rotation_{ROTATIONS[i]}_exp_{i}" for i in range(len(ROTATIONS))
        ]

    print(json.dumps({k: v["summary"] for k, v in conditions.items()}, indent=2))
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
