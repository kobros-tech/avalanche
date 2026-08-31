# Skill Memory benchmark

This document describes two development benchmarks for the Skill Memory integration. The arithmetic benchmark is a controlled proof-of-benefit experiment; the RotatedMNIST benchmark is the first standard Avalanche scenario used to test whether the behavior extends beyond arithmetic.

## Controlled arithmetic benchmark

The arithmetic benchmark compares two otherwise identical Avalanche `Naive` runs:

- **baseline**: normal sequential training with no Skill Memory plugin;
- **skill_memory**: the same model, optimizer, data, seeds, batch size, and epoch budget,
  with `SkillMemoryPlugin` enabled.

The arithmetic compatibility function is deliberately explicit and is not intended to be a learned task-similarity model. Evaluation data are independent from training data and are never passed to the compatibility function.

For the 15-seed benchmark, the output includes final MSE, MSE history, forgetting, decisions, stored skills, aggregate mean/std/95% CI, and paired per-seed differences between Skill Memory and baseline.

Forgetting is defined as `max(0, final MSE - minimum MSE observed after any experience for that task)`. The final task is excluded from forgetting because there is no later experience after which forgetting can be measured.

The arithmetic output is written to `results/skill_memory_benchmark_multiseed.json`.

## Standard Avalanche benchmark: RotatedMNIST

The next-stage benchmark uses Avalanche's standard `RotatedMNIST` scenario rather than the synthetic arithmetic stream. The transformation sequence is:

| Experience | Rotation | Expected Skill Memory decision |
| --- | ---: | --- |
| 0 | 0° | acquire |
| 1 | 30° | acquire |
| 2 | 60° | acquire |
| 3 | 0° | reuse the earlier 0° skill |
| 4 | 30° | reuse the earlier 30° skill |

Repeated transformations provide legitimate reuse opportunities without hard-coding source/target task pairs. Compatibility is based only on the known training-task rotation metadata; test samples and targets are not used for the reuse decision.

Three conditions are compared:

- **naive**: standard Avalanche `Naive` training;
- **replay**: `Naive` plus Avalanche `ReplayPlugin` with a fixed replay memory;
- **skill_memory**: `Naive` plus `SkillMemoryPlugin`.

The standard benchmark uses three fixed seeds for the initial Step 2 validation. It records per-experience test accuracy, final accuracy, forgetting on previously seen experiences, Skill Memory acquisition/reuse decisions, stored skills, and training time.

This benchmark is deliberately modest: it establishes whether the prototype can operate in a standard Avalanche continual-learning scenario and provides a comparison against replay. It is not yet the final upstream-quality benchmark. If the implementation is stable and the effect is meaningful, the next iteration should expand the seed count and add more standard scenarios before making broad claims.

The standard benchmark output is written to `results/skill_memory_rotated_mnist.json`.

## Interpretation

The arithmetic experiment is the controlled mechanistic test. The RotatedMNIST experiment is the first generalization test. A positive result on either benchmark should be interpreted in context; neither benchmark alone establishes general continual-learning superiority.

## Running

From the repository root:

```bash
python examples/skill_memory_benchmark.py
python examples/skill_memory_standard_benchmark.py
```
