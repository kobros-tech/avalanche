"""Standard Avalanche benchmark for Skill Memory.

This experiment moves beyond the synthetic arithmetic proof of concept and uses
Avalanche's standard RotatedMNIST benchmark. The sequence intentionally repeats
transformations (0, 30, 60, 0, 30 degrees), creating legitimate opportunities
for reuse without hard-coding source/target task pairs.

Compatibility is derived only from the known training-task transformation
metadata. Test samples are never used to choose a skill.

Conditions:
* naive: ordinary Avalanche Naive strategy.
* replay: Naive + Avalanche ReplayPlugin.
* skill_memory: Naive + SkillMemoryPlugin.

The experiment reports per-experience test accuracy, forgetting on previously
seen experiences, reuse/acquisition decisions, and training time.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from avalanche.benchmarks.classic import RotatedMNIST
from avalanche.models import SimpleMLP
from avalanche.training import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.skill_memory import SkillMemory, SkillMemoryPlugin


SEEDS = [0, 1, 2]
ROTATIONS = [0, 30, 60, 0, 30]
EPOCHS = 1
TRAIN_MB_SIZE = 128
EVAL_MB_SIZE = 256
REPLAY_MEM_SIZE = 200
MEMORY_MAX_SKILLS = 5
REUSE_THRESHOLD = 1.0


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


def build_strategy(condition: str):
    model = SimpleMLP(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    plugins = []
    memory = None
    skill_plugin = None

    if condition == "replay":
        plugins.append(ReplayPlugin(mem_size=REPLAY_MEM_SIZE))
    elif condition == "skill_memory":
        memory = SkillMemory(max_skills=MEMORY_MAX_SKILLS)

        def compatibility(record, query):
            """Score a stored skill against an Avalanche experience.

            ``SkillMemoryPlugin`` passes the actual Avalanche experience as the
            query. The experiment's task descriptor is known from the training
            benchmark definition, so compatibility can derive the rotation
            without relying on test data or converting the experience to a
            dictionary.
            """
            query_rotation = ROTATIONS[query.current_experience]
            return 1.0 if record.metadata["rotation"] == query_rotation else 0.0

        def metadata(exp_index: int):
            return {"rotation": ROTATIONS[exp_index]}

        skill_plugin = SkillMemoryPlugin(
            memory=memory,
            skill_name=lambda exp: (
                f"rotation_{ROTATIONS[exp.current_experience]}_exp_{exp.current_experience}"
            ),
            skill_metadata=lambda exp: metadata(exp.current_experience),
            compatibility=compatibility,
            threshold=REUSE_THRESHOLD,
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
    strategy, memory, plugin = build_strategy(condition)

    accuracy_history: List[Dict[str, float]] = []
    decisions: List[Dict[str, object]] = []
    train_seconds = 0.0

    for experience in benchmark.train_stream:
        started = time.perf_counter()
        strategy.train(experience, eval_streams=[])
        train_seconds += time.perf_counter() - started

        exp_id = experience.current_experience
        if condition != "skill_memory":
            decision = "baseline" if condition == "naive" else "replay"
            score = None
        elif plugin.last_reused_skill is None:
            decision = "acquire"
            score = plugin.last_compatibility_score
        else:
            decision = f"reuse:{plugin.last_reused_skill}"
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
        for test_index, test_experience in enumerate(benchmark.test_stream):
            if test_index > exp_id:
                break
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
    for seed in SEEDS:
        print(f"Running standard benchmark seed {seed}...")
        for condition in ("naive", "replay", "skill_memory"):
            print(f"  condition={condition}")
            all_results.append(run_condition(condition, seed))

    conditions = {}
    for condition in ("naive", "replay", "skill_memory"):
        runs = [r for r in all_results if r["condition"] == condition]
        conditions[condition] = {"runs": runs, "summary": summarize(runs)}

    result = {
        "benchmark": "skill_memory_rotated_mnist_v1",
        "description": (
            "Standard Avalanche RotatedMNIST benchmark comparing Naive, Replay, "
            "and Skill Memory. Repeated rotations create non-arithmetic reuse opportunities."
        ),
        "seeds": SEEDS,
        "rotations": ROTATIONS,
        "conditions": conditions,
    }

    output_path = Path("results/skill_memory_rotated_mnist.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for run in conditions["skill_memory"]["runs"]:
        decisions = [item["decision"] for item in run["decisions"]]
        assert decisions == [
            "acquire",
            "acquire",
            "acquire",
            "reuse:rotation_0_exp_0",
            "reuse:rotation_30_exp_1",
        ]

    print(json.dumps({k: v["summary"] for k, v in conditions.items()}, indent=2))
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
