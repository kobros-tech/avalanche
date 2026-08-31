"""Controlled multi-seed benchmark for the Skill Memory Avalanche integration.

The benchmark compares otherwise identical continual-learning runs:

* ``baseline``: ordinary Naive training across four arithmetic tasks.
* ``skill_memory``: the same training with SkillMemoryPlugin enabled.

This benchmark is intended to quantify whether the observed single-seed result
is robust across fixed seeds. It reports per-seed metrics plus aggregate mean,
standard deviation, and a 95% confidence interval for final MSE and forgetting.
The compatibility function uses only training-experience descriptors; evaluation
samples are generated independently and never passed to the reuse decision.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

from avalanche.training import Naive
from avalanche.training.skill_memory import SkillMemory, SkillMemoryPlugin


SEEDS = list(range(15))
TRAIN_SAMPLES = 64
EVAL_SAMPLES = 64
TRAIN_SEED_BASE = 100
EVAL_SEED_BASE = 10_000
EPOCHS = 3
TASKS = ["multiply", "add", "square", "divide"]
FORGETTING_TASKS = TASKS[:-1]


@dataclass
class ArithmeticStream:
    name: str


class ArithmeticDataset(Dataset):
    def __init__(self, tensors: Tuple[torch.Tensor, torch.Tensor]):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        return self.tensors[0][index], self.tensors[1][index]

    def train(self):
        return self

    def eval(self):
        return self


@dataclass
class ArithmeticExperience:
    current_experience: int
    operation: str
    prerequisites: Tuple[str, ...]
    dataset: ArithmeticDataset
    origin_stream: ArithmeticStream

    def logging(self):
        return self


def make_dataset(operation: str, n_samples: int, seed: int) -> ArithmeticDataset:
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n_samples, 2, generator=generator) * 4.0 + 1.0
    a, b = x[:, 0], x[:, 1]

    if operation == "multiply":
        y = a * b
    elif operation == "add":
        y = a + b
    elif operation == "square":
        y = a.square()
    elif operation == "divide":
        y = a / b
    else:
        raise ValueError(f"unknown operation: {operation}")

    return ArithmeticDataset((x, y.unsqueeze(1)))


def make_experience(index: int, operation: str, prerequisites: Tuple[str, ...], seed: int):
    return ArithmeticExperience(
        current_experience=index,
        operation=operation,
        prerequisites=prerequisites,
        dataset=make_dataset(operation, TRAIN_SAMPLES, TRAIN_SEED_BASE + seed * len(TASKS) + index),
        origin_stream=ArithmeticStream(name="arithmetic_train"),
    )


def compatibility(record, experience: ArithmeticExperience) -> float:
    return float(record.metadata.get("operation") in experience.prerequisites)


def evaluate(model: nn.Module, operation: str, seed: int) -> float:
    dataset = make_dataset(operation, EVAL_SAMPLES, EVAL_SEED_BASE + seed * len(TASKS) + TASKS.index(operation))
    model.eval()
    with torch.no_grad():
        predictions = model(dataset.tensors[0])
        return float(nn.functional.mse_loss(predictions, dataset.tensors[1]))


def build_strategy(use_skill_memory: bool):
    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    plugins = []
    memory = None
    skill_plugin = None

    if use_skill_memory:
        memory = SkillMemory(max_skills=8)

        def metadata(exp: ArithmeticExperience):
            return {
                "operation": exp.operation,
                "prerequisites": list(exp.prerequisites),
            }

        skill_plugin = SkillMemoryPlugin(
            memory=memory,
            skill_name=lambda exp: exp.operation,
            skill_metadata=metadata,
            compatibility=compatibility,
            threshold=0.5,
        )
        plugins.append(skill_plugin)

    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        train_mb_size=16,
        train_epochs=EPOCHS,
        eval_mb_size=16,
        eval_every=-1,
        plugins=plugins,
    )
    return strategy, memory, skill_plugin


def run_condition(use_skill_memory: bool, seed: int) -> Dict:
    torch.manual_seed(seed)
    strategy, memory, plugin = build_strategy(use_skill_memory)

    experiences = [
        make_experience(0, "multiply", (), seed),
        make_experience(1, "add", (), seed),
        make_experience(2, "square", ("multiply",), seed),
        make_experience(3, "divide", (), seed),
    ]

    mse_history: List[Dict[str, float]] = []
    decisions: List[Dict[str, object]] = []

    for experience in experiences:
        strategy.train(experience, eval_streams=[])

        if plugin is None:
            decision = "baseline"
            score = None
        elif plugin.last_reused_skill is None:
            decision = "acquire"
            score = plugin.last_compatibility_score
        else:
            decision = f"reuse:{plugin.last_reused_skill}"
            score = plugin.last_compatibility_score

        decisions.append({
            "operation": experience.operation,
            "decision": decision,
            "compatibility_score": score,
        })

        current = {
            operation: evaluate(strategy.model, operation, seed)
            for index, operation in enumerate(TASKS)
            if index <= experience.current_experience
        }
        mse_history.append(current)

    final_mse = mse_history[-1]
    forgetting = {}
    for operation in FORGETTING_TASKS:
        values = [row[operation] for row in mse_history if operation in row]
        forgetting[operation] = max(0.0, values[-1] - min(values))

    return {
        "condition": "skill_memory" if use_skill_memory else "baseline",
        "seed": seed,
        "epochs_per_experience": EPOCHS,
        "train_samples_per_experience": TRAIN_SAMPLES,
        "eval_samples_per_task": EVAL_SAMPLES,
        "mse_history": mse_history,
        "final_mse": final_mse,
        "forgetting_mse": forgetting,
        "decisions": decisions,
        "stored_skills": memory.names() if memory is not None else [],
    }


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def ci95(values: List[float]) -> List[float]:
    average = mean(values)
    half_width = 1.96 * std(values) / math.sqrt(len(values))
    return [average - half_width, average + half_width]


def summarize(runs: List[Dict]) -> Dict:
    summary = {}
    for task in TASKS:
        values = [run["final_mse"][task] for run in runs]
        summary[task] = {
            "mean": mean(values),
            "std": std(values),
            "ci95": ci95(values),
        }

    summary["forgetting_mse"] = {}
    for task in FORGETTING_TASKS:
        values = [run["forgetting_mse"][task] for run in runs]
        summary["forgetting_mse"][task] = {
            "mean": mean(values),
            "std": std(values),
            "ci95": ci95(values),
        }

    return summary


def paired_metric_summary(
    baseline_runs: List[Dict], skill_memory_runs: List[Dict], metric: str
) -> Dict:
    """Summarize paired Skill Memory minus baseline differences by task."""
    tasks = TASKS if metric == "final_mse" else FORGETTING_TASKS
    result = {}
    for task in tasks:
        if metric == "final_mse":
            baseline_values = [run["final_mse"][task] for run in baseline_runs]
            skill_values = [run["final_mse"][task] for run in skill_memory_runs]
        elif metric == "forgetting_mse":
            baseline_values = [run["forgetting_mse"][task] for run in baseline_runs]
            skill_values = [run["forgetting_mse"][task] for run in skill_memory_runs]
        else:
            raise ValueError(f"unknown metric: {metric}")

        differences = [skill - baseline for skill, baseline in zip(skill_values, baseline_values)]
        wins = sum(difference < 0.0 for difference in differences)
        losses = sum(difference > 0.0 for difference in differences)
        ties = len(differences) - wins - losses
        result[task] = {
            "skill_memory_minus_baseline_mean": mean(differences),
            "skill_memory_minus_baseline_std": std(differences),
            "ci95": ci95(differences),
            "skill_memory_wins": wins,
            "baseline_wins": losses,
            "ties": ties,
            "n": len(differences),
        }
    return result


def main() -> None:
    results = []
    for seed in SEEDS:
        print(f"Running seed {seed}...")
        baseline = run_condition(False, seed)
        skill_memory = run_condition(True, seed)
        results.append({"seed": seed, "baseline": baseline, "skill_memory": skill_memory})

    baseline_runs = [item["baseline"] for item in results]
    skill_memory_runs = [item["skill_memory"] for item in results]

    result = {
        "benchmark": "skill_memory_arithmetic_multiseed_v2",
        "description": "15-seed controlled baseline vs Skill Memory continual-learning benchmark.",
        "seeds": SEEDS,
        "metric_definitions": {
            "final_mse": "MSE on an independent evaluation set after the final experience.",
            "forgetting_mse": "max(0, final MSE - minimum MSE observed after any experience for that task).",
            "paired_difference": "Skill Memory metric minus baseline metric for the same seed.",
            "confidence_interval": "Approximate 95% CI using mean +/- 1.96 * sample SD / sqrt(n).",
            "forgetting_scope": "Only previously encountered tasks (multiply, add, square); divide is excluded because there is no later experience after it to measure forgetting.",
        },
        "conditions": {
            "baseline": {
                "runs": baseline_runs,
                "summary": summarize(baseline_runs),
            },
            "skill_memory": {
                "runs": skill_memory_runs,
                "summary": summarize(skill_memory_runs),
            },
        },
        "paired_comparison": {
            "final_mse": paired_metric_summary(baseline_runs, skill_memory_runs, "final_mse"),
            "forgetting_mse": paired_metric_summary(baseline_runs, skill_memory_runs, "forgetting_mse"),
        },
    }

    output_path = Path("results/skill_memory_benchmark_multiseed.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for condition_name in ("baseline", "skill_memory"):
        print(f"\n[{condition_name}]")
        print(json.dumps(result["conditions"][condition_name]["summary"], indent=2))

    print("\n[paired final MSE]")
    print(json.dumps(result["paired_comparison"]["final_mse"], indent=2))
    print("\n[paired forgetting MSE]")
    print(json.dumps(result["paired_comparison"]["forgetting_mse"], indent=2))

    for run in skill_memory_runs:
        decisions = [item["decision"] for item in run["decisions"]]
        assert decisions == ["acquire", "acquire", "reuse:multiply", "acquire"]
        assert run["stored_skills"] == TASKS

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
