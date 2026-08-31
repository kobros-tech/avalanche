"""Controlled benchmark for the Skill Memory Avalanche integration.

The benchmark compares two otherwise identical continual-learning runs:

* ``baseline``: ordinary Naive training across the four arithmetic tasks.
* ``skill_memory``: the same training with SkillMemoryPlugin enabled.

The task sequence is deliberately small and controlled. Multiplication and
addition are acquired first; squaring declares multiplication as a prerequisite
and can therefore reuse it; division declares an incompatible prerequisite and
must be acquired normally.

Evaluation data are generated independently from training data and are never
passed to the compatibility function. The benchmark is intended to provide
compact evidence for whether the integration is worth further investigation,
not to establish a general research claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

from avalanche.training import Naive
from avalanche.training.skill_memory import SkillMemory, SkillMemoryPlugin


SEED = 7
TRAIN_SAMPLES = 64
EVAL_SAMPLES = 64
TRAIN_SEED_BASE = 100
EVAL_SEED_BASE = 10_000
EPOCHS = 3


@dataclass
class ArithmeticStream:
    """Minimal stream descriptor required by Avalanche's training template."""

    name: str


class ArithmeticDataset(Dataset):
    """Small dataset adapter implementing Avalanche's train/eval protocol."""

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
    """Minimal Avalanche-compatible supervised experience for the benchmark."""

    current_experience: int
    operation: str
    prerequisites: Tuple[str, ...]
    dataset: ArithmeticDataset
    origin_stream: ArithmeticStream

    def logging(self):
        """Return the representation expected by EvaluationPlugin."""
        return self


def make_dataset(operation: str, n_samples: int, seed: int) -> ArithmeticDataset:
    """Generate deterministic data for one arithmetic operation."""
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


def make_experience(
    index: int, operation: str, prerequisites: Tuple[str, ...]
) -> ArithmeticExperience:
    """Create one training experience."""
    return ArithmeticExperience(
        current_experience=index,
        operation=operation,
        prerequisites=prerequisites,
        dataset=make_dataset(operation, TRAIN_SAMPLES, TRAIN_SEED_BASE + index),
        origin_stream=ArithmeticStream(name="arithmetic_train"),
    )


def compatibility(record, experience: ArithmeticExperience) -> float:
    """Return compatibility from training-task descriptors only."""
    return float(record.metadata.get("operation") in experience.prerequisites)


def evaluate(model: nn.Module, operation: str, experience_index: int) -> float:
    """Evaluate on an independent, deterministic dataset."""
    dataset = make_dataset(operation, EVAL_SAMPLES, EVAL_SEED_BASE + experience_index)
    model.eval()
    with torch.no_grad():
        predictions = model(dataset.tensors[0])
        return float(nn.functional.mse_loss(predictions, dataset.tensors[1]))


def build_strategy(use_skill_memory: bool):
    """Build identical training strategies, optionally adding Skill Memory."""
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


def run_condition(use_skill_memory: bool) -> Dict:
    """Run one benchmark condition from an identical initialization."""
    torch.manual_seed(SEED)
    strategy, memory, plugin = build_strategy(use_skill_memory)

    experiences = [
        make_experience(0, "multiply", ()),
        make_experience(1, "add", ()),
        make_experience(2, "square", ("multiply",)),
        make_experience(3, "divide", ("division",)),
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

        decisions.append(
            {
                "operation": experience.operation,
                "decision": decision,
                "compatibility_score": score,
            }
        )

        current = {
            operation: evaluate(strategy.model, operation, index)
            for index, operation in enumerate(
                ["multiply", "add", "square", "divide"]
            )
            if index <= experience.current_experience
        }
        mse_history.append(current)

    final_mse = mse_history[-1]
    forgetting = {}
    for operation in ("multiply", "add", "square"):
        values = [row[operation] for row in mse_history if operation in row]
        forgetting[operation] = max(0.0, values[-1] - min(values))

    return {
        "condition": "skill_memory" if use_skill_memory else "baseline",
        "seed": SEED,
        "epochs_per_experience": EPOCHS,
        "train_samples_per_experience": TRAIN_SAMPLES,
        "eval_samples_per_task": EVAL_SAMPLES,
        "mse_history": mse_history,
        "final_mse": final_mse,
        "forgetting_mse": forgetting,
        "decisions": decisions,
        "stored_skills": memory.names() if memory is not None else [],
    }


def main() -> None:
    """Run both conditions and write machine-readable results."""
    baseline = run_condition(False)
    skill_memory = run_condition(True)

    result = {
        "benchmark": "skill_memory_arithmetic_v1",
        "description": "Controlled baseline vs Skill Memory continual-learning run.",
        "seed": SEED,
        "conditions": [baseline, skill_memory],
    }

    output_path = Path("results/skill_memory_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("Skill Memory benchmark")
    for condition in result["conditions"]:
        print(f"\n[{condition['condition']}]")
        print(f"final_mse={condition['final_mse']}")
        print(f"forgetting_mse={condition['forgetting_mse']}")
        print(f"decisions={condition['decisions']}")

    assert skill_memory["decisions"][0]["decision"] == "acquire"
    assert skill_memory["decisions"][1]["decision"] == "acquire"
    assert skill_memory["decisions"][2]["decision"] == "reuse:multiply"
    assert skill_memory["decisions"][3]["decision"] == "acquire"
    assert skill_memory["stored_skills"] == [
        "multiply",
        "add",
        "square",
        "divide",
    ]

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
