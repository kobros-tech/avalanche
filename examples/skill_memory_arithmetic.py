"""Arithmetic proof of concept for Skill Memory."""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

from avalanche.training import Naive
from avalanche.training.skill_memory import SkillMemory, SkillMemoryPlugin


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
    """Minimal Avalanche-compatible supervised experience for this PoC."""

    current_experience: int
    operation: str
    prerequisites: Tuple[str, ...]
    dataset: ArithmeticDataset
    origin_stream: ArithmeticStream

    def logging(self):
        """Return the experience representation expected by EvaluationPlugin."""
        return self


def make_dataset(
    operation: str, n_samples: int, seed: int
) -> ArithmeticDataset:
    """Create deterministic train/evaluation data for one arithmetic operation."""
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
    """Build one deterministic arithmetic experience."""
    return ArithmeticExperience(
        current_experience=index,
        operation=operation,
        prerequisites=prerequisites,
        dataset=make_dataset(operation, n_samples=64, seed=100 + index),
        origin_stream=ArithmeticStream(name="arithmetic_train"),
    )


def compatibility(record, experience: ArithmeticExperience) -> float:
    """Score compatibility using only stored metadata and task descriptors."""
    operation = record.metadata.get("operation")
    return 1.0 if operation in experience.prerequisites else 0.0


def evaluate(model: nn.Module, experience: ArithmeticExperience) -> float:
    """Compute MSE on an independent evaluation set."""
    eval_data = make_dataset(
        experience.operation,
        n_samples=64,
        seed=10_000 + experience.current_experience,
    )
    model.eval()
    with torch.no_grad():
        predictions = model(eval_data.tensors[0])
        return float(nn.functional.mse_loss(predictions, eval_data.tensors[1]))


def main() -> None:
    """Run the four-stage arithmetic Skill Memory proof of concept."""
    torch.manual_seed(7)

    experiences = [
        make_experience(0, "multiply", ()),
        make_experience(1, "add", ()),
        make_experience(2, "square", ("multiply",)),
        make_experience(3, "divide", ("division",)),
    ]

    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    memory = SkillMemory(max_skills=8)
    decisions: Dict[str, str] = {}

    def name(exp: ArithmeticExperience) -> str:
        return exp.operation

    def metadata(exp: ArithmeticExperience):
        return {
            "operation": exp.operation,
            "prerequisites": list(exp.prerequisites),
        }

    plugin = SkillMemoryPlugin(
        memory=memory,
        skill_name=name,
        skill_metadata=metadata,
        compatibility=compatibility,
        threshold=0.5,
    )

    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        train_mb_size=16,
        train_epochs=3,
        eval_mb_size=16,
        eval_every=-1,
        plugins=[plugin],
    )

    for experience in experiences:
        strategy.train(experience, eval_streams=[])
        decision = (
            f"reuse:{plugin.last_reused_skill}"
            if plugin.last_reused_skill is not None
            else "acquire"
        )
        decisions[experience.operation] = decision
        mse = evaluate(model, experience)
        print(
            f"experience={experience.current_experience} "
            f"operation={experience.operation} "
            f"decision={decision} "
            f"score={plugin.last_compatibility_score:.3f} "
            f"eval_mse={mse:.6f}"
        )

    assert decisions["multiply"] == "acquire"
    assert decisions["add"] == "acquire"
    assert decisions["square"] == "reuse:multiply"
    assert decisions["divide"] == "acquire"
    assert memory.contains("multiply")
    assert memory.contains("add")
    assert memory.contains("square")
    assert memory.contains("divide")

    print(f"stored_skills={memory.names()}")
    print("Arithmetic Skill Memory PoC completed successfully.")


if __name__ == "__main__":
    main()
