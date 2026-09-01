"""Small demonstration of the evidence-based CLONE-vs-SCRATCH policy.

The example deliberately avoids the old fixed 0.30 clone threshold. A prior
skill is cloned only when a matched, small adaptation probe shows lower
post-adaptation loss than a fresh model. REUSE still requires strong zero-shot
compatibility because REUSE does not adapt the source skill.
"""

import torch
from torch import nn

from avalanche.training import Naive
from avalanche.training.skill_memory import (
    AdaptationCompatibilityScorer,
    ProbeCompatibilityScorer,
    SkillMemory,
    SkillMemoryPlugin,
    mean_baseline_mse,
)
from skill_memory_benchmark import make_dataset, make_model, make_experience


PROBE_SAMPLES = 16


def probe(experience, seed=9000):
    data = make_dataset(experience.operation, PROBE_SAMPLES, seed + experience.current_experience)
    return data.tensors


def main():
    torch.manual_seed(7)
    memory = SkillMemory(max_skills=8)

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    zero_shot = ProbeCompatibilityScorer(
        model_factory=make_model,
        loss_fn=nn.functional.mse_loss,
        probe_fn=probe,
        reference_fn=lambda exp: mean_baseline_mse(probe(exp)[1]),
    )
    adaptation = AdaptationCompatibilityScorer(
        model_factory=make_model,
        loss_fn=nn.functional.mse_loss,
        probe_fn=probe,
        steps=3,
        optimizer_factory=lambda parameters: torch.optim.Adam(parameters, lr=1e-2),
    )

    plugin = SkillMemoryPlugin(
        memory=memory,
        skill_name=lambda exp: exp.operation,
        skill_metadata=lambda exp: {"operation": exp.operation},
        compatibility=zero_shot,
        clone_compatibility=adaptation,
        reuse_threshold=0.90,
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

    for index, operation in enumerate(("multiply", "square", "divide")):
        experience = make_experience(index, operation, (), seed=7)
        strategy.train(experience, eval_streams=[])
        print(
            f"task={operation} decision={plugin.last_decision} "
            f"source={plugin.last_selected_skill} "
            f"zero_shot={plugin.last_compatibility_score:.3f} "
            f"clone_value={plugin.last_clone_value:.3f}"
        )

    print(f"stored_skills={memory.names()}")


if __name__ == "__main__":
    main()
