"""Controlled multi-seed benchmark for the Skill Memory Avalanche integration."""

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
from avalanche.training.skill_memory import (
    AdaptationCompatibilityScorer,
    ProbeCompatibilityScorer,
    SkillMemory,
    SkillMemoryPlugin,
    mean_baseline_mse,
)


SEEDS = list(range(15))
TRAIN_SAMPLES = 64
EVAL_SAMPLES = 64
PROBE_SAMPLES = 16
ADAPTATION_SAMPLES = 64
TRAIN_SEED_BASE = 100
EVAL_SEED_BASE = 10_000
PROBE_SEED_BASE = 50_000
ADAPTATION_SEED_BASE = 60_000
EPOCHS = 3
TRAIN_MB_SIZE = 16
TASKS = ["multiply", "add", "square", "divide"]
FORGETTING_TASKS = TASKS[:-1]
REUSE_THRESHOLD = 0.90
# Match the actual Avalanche learner exactly: 64 training samples, minibatch
# size 16, 3 epochs = 4 deterministic minibatches/epoch = 12 updates.
ADAPTATION_BATCH_SIZE = TRAIN_MB_SIZE
ADAPTATION_STEPS = EPOCHS * (TRAIN_SAMPLES // TRAIN_MB_SIZE)


def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))


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


def evaluate(model: nn.Module, operation: str, seed: int) -> float:
    dataset = make_dataset(operation, EVAL_SAMPLES, EVAL_SEED_BASE + seed * len(TASKS) + TASKS.index(operation))
    model.eval()
    with torch.no_grad():
        return float(nn.functional.mse_loss(model(dataset.tensors[0]), dataset.tensors[1]))


def probe_batch(experience: ArithmeticExperience, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Small zero-shot probe used only for REUSE compatibility."""
    dataset = make_dataset(
        operation=experience.operation,
        n_samples=PROBE_SAMPLES,
        seed=PROBE_SEED_BASE + seed * len(TASKS) + experience.current_experience,
    )
    return dataset.tensors


def adaptation_data(experience: ArithmeticExperience, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fresh 64-sample adaptation data for the CLONE-vs-SCRATCH scorer."""
    dataset = make_dataset(
        operation=experience.operation,
        n_samples=ADAPTATION_SAMPLES,
        seed=ADAPTATION_SEED_BASE + seed * len(TASKS) + experience.current_experience,
    )
    return dataset.tensors


def build_strategy(use_skill_memory: bool, seed: int, force_decision: str = None):
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    plugins = []
    memory = None
    skill_plugin = None

    if use_skill_memory:
        memory = SkillMemory(max_skills=8)

        compatibility = ProbeCompatibilityScorer(
            model_factory=make_model,
            loss_fn=nn.functional.mse_loss,
            probe_fn=lambda exp: probe_batch(exp, seed),
            reference_fn=lambda exp: mean_baseline_mse(probe_batch(exp, seed)[1]),
        )
        clone_compatibility = AdaptationCompatibilityScorer(
            model_factory=make_model,
            loss_fn=nn.functional.mse_loss,
            probe_fn=lambda exp: probe_batch(exp, seed),
            adaptation_fn=lambda exp: adaptation_data(exp, seed),
            batch_size=ADAPTATION_BATCH_SIZE,
            steps=ADAPTATION_STEPS,
            optimizer_factory=lambda parameters: torch.optim.Adam(parameters, lr=1e-2),
        )

        skill_plugin = SkillMemoryPlugin(
            memory=memory,
            skill_name=lambda exp: exp.operation,
            skill_metadata=lambda exp: {
                "operation": exp.operation,
                "prerequisites": list(exp.prerequisites),
            },
            compatibility=compatibility,
            clone_compatibility=clone_compatibility,
            reuse_threshold=REUSE_THRESHOLD,
            force_decision=force_decision,
        )
        plugins.append(skill_plugin)

    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        train_mb_size=TRAIN_MB_SIZE,
        train_epochs=EPOCHS,
        eval_mb_size=16,
        eval_every=-1,
        plugins=plugins,
    )
    return strategy, memory, skill_plugin


def run_condition(use_skill_memory: bool, seed: int, force_decision: str = None) -> Dict:
    torch.manual_seed(seed)
    strategy, memory, plugin = build_strategy(use_skill_memory, seed, force_decision=force_decision)
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
        decision = "baseline" if plugin is None else plugin.last_decision
        if plugin is not None and plugin.last_selected_skill is not None:
            decision = f"{decision}:{plugin.last_selected_skill}"
        decisions.append({
            "operation": experience.operation,
            "decision": decision,
            "compatibility_score": None if plugin is None else plugin.last_compatibility_score,
            "clone_value": None if plugin is None else plugin.last_clone_value,
        })
        mse_history.append({
            operation: evaluate(strategy.model, operation, seed)
            for index, operation in enumerate(TASKS) if index <= experience.current_experience
        })

    final_mse = mse_history[-1]
    forgetting = {}
    for operation in FORGETTING_TASKS:
        values = [row[operation] for row in mse_history if operation in row]
        forgetting[operation] = max(0.0, values[-1] - min(values))

    return {
        "condition": "skill_memory" if use_skill_memory else "baseline",
        "seed": seed,
        "epochs_per_experience": EPOCHS,
        "train_mb_size": TRAIN_MB_SIZE,
        "adaptation_samples_for_clone_decision": ADAPTATION_SAMPLES,
        "adaptation_batch_size_for_clone_decision": ADAPTATION_BATCH_SIZE,
        "adaptation_steps_for_clone_decision": ADAPTATION_STEPS,
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
        summary[task] = {"mean": mean(values), "std": std(values), "ci95": ci95(values)}
    summary["forgetting_mse"] = {}
    for task in FORGETTING_TASKS:
        values = [run["forgetting_mse"][task] for run in runs]
        summary["forgetting_mse"][task] = {"mean": mean(values), "std": std(values), "ci95": ci95(values)}
    return summary


def paired_metric_summary(baseline_runs: List[Dict], skill_memory_runs: List[Dict], metric: str) -> Dict:
    tasks = TASKS if metric == "final_mse" else FORGETTING_TASKS
    result = {}
    for task in tasks:
        baseline_values = [run[metric][task] for run in baseline_runs]
        skill_values = [run[metric][task] for run in skill_memory_runs]
        differences = [skill - baseline for skill, baseline in zip(skill_values, baseline_values)]
        wins = sum(difference < 0.0 for difference in differences)
        losses = sum(difference > 0.0 for difference in differences)
        result[task] = {
            "skill_memory_minus_baseline_mean": mean(differences),
            "skill_memory_minus_baseline_std": std(differences),
            "ci95": ci95(differences),
            "skill_memory_wins": wins,
            "baseline_wins": losses,
            "ties": len(differences) - wins - losses,
            "n": len(differences),
        }
    return result


def oracle_agreement(skill_memory_runs: List[Dict], oracle_runs: Dict[str, List[Dict]]) -> Dict:
    result = {}
    for task_index, task in enumerate(TASKS):
        policy = [run["decisions"][task_index]["decision"].split(":")[0] for run in skill_memory_runs]
        oracle_choices = []
        for seed_index in range(len(SEEDS)):
            candidates = {
                condition: oracle_runs[condition][seed_index]["final_mse"][task]
                for condition in ("reuse", "clone", "scratch")
            }
            oracle_choices.append(min(candidates, key=candidates.get))
        matches = sum(a == b for a, b in zip(policy, oracle_choices))
        result[task] = {"matches": matches, "n": len(SEEDS), "agreement": matches / len(SEEDS)}
    return result


def main() -> None:
    results = []
    for seed in SEEDS:
        print(f"Running seed {seed}...")
        results.append({
            "seed": seed,
            "baseline": run_condition(False, seed),
            "skill_memory": run_condition(True, seed),
            "reuse": run_condition(True, seed, force_decision=SkillMemoryPlugin.REUSE),
            "clone": run_condition(True, seed, force_decision=SkillMemoryPlugin.CLONE),
            "scratch": run_condition(True, seed, force_decision=SkillMemoryPlugin.SCRATCH),
        })

    baseline_runs = [item["baseline"] for item in results]
    skill_runs = [item["skill_memory"] for item in results]
    oracle_runs = {condition: [item[condition] for item in results] for condition in ("reuse", "clone", "scratch")}

    result = {
        "benchmark": "skill_memory_arithmetic_multiseed_v6_matched_adaptation",
        "description": "15-seed baseline plus oracle control benchmark. CLONE scoring now adapts candidate and SCRATCH models on the same 64 fresh training-distribution samples using batch size 16 for exactly 12 deterministic optimizer updates, matching the Avalanche learner's 3 epochs of 4 minibatches.",
        "policy": {
            "reuse_threshold": REUSE_THRESHOLD,
            "clone_threshold": None,
            "clone_rule": "select the stored candidate with the greatest positive post-adaptation improvement over a fresh model; otherwise SCRATCH",
            "adaptation_samples": ADAPTATION_SAMPLES,
            "adaptation_batch_size": ADAPTATION_BATCH_SIZE,
            "adaptation_steps": ADAPTATION_STEPS,
            "adaptation_epochs_equivalent": EPOCHS,
            "probe": "fresh samples from the new experience training distribution; never the evaluation stream",
        },
        "seeds": SEEDS,
        "conditions": {
            "baseline": {"runs": baseline_runs, "summary": summarize(baseline_runs)},
            "skill_memory": {"runs": skill_runs, "summary": summarize(skill_runs)},
            "oracle_reuse": {"summary": summarize(oracle_runs["reuse"])},
            "oracle_clone": {"summary": summarize(oracle_runs["clone"])},
            "oracle_scratch": {"summary": summarize(oracle_runs["scratch"])},
        },
        "paired_comparison": {
            "final_mse": paired_metric_summary(baseline_runs, skill_runs, "final_mse"),
            "forgetting_mse": paired_metric_summary(baseline_runs, skill_runs, "forgetting_mse"),
        },
        "policy_oracle_agreement": oracle_agreement(skill_runs, oracle_runs),
    }

    output_path = Path("results/skill_memory_benchmark_multiseed.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for run in skill_runs:
        assert run["decisions"][0]["decision"] == "scratch"
        assert run["stored_skills"] == TASKS

    print(json.dumps(result["paired_comparison"], indent=2))
    print(json.dumps(result["policy_oracle_agreement"], indent=2))
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
