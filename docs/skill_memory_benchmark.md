# Skill Memory benchmark

This document describes three development benchmarks for the Skill Memory integration. The arithmetic benchmark is a controlled proof-of-benefit experiment; the policy oracle benchmark asks the sharper question of whether the automatic decision is actually the *right* one; the RotatedMNIST benchmark is the first standard Avalanche scenario used to test whether the behavior extends beyond arithmetic.

## Why probe-based compatibility

Earlier versions of both the arithmetic and RotatedMNIST benchmarks scored compatibility with a hand-written binary lookup: "is this operation a declared prerequisite?" or "does the rotation match exactly?". Both can only ever return exactly `0.0` or `1.0`.

`avalanche/training/skill_memory/compatibility.py` adds `ProbeCompatibilityScorer`, a single, continuous, generic replacement: load the candidate skill into a fresh copy of the model, measure its **zero-shot** error on a small probe drawn from the new experience's own training data (never the held-out evaluation/test stream), and compare that error against a cheap reference (a mean predictor for regression via `mean_baseline_mse`, or a uniform-class guess for classification via `uniform_guess_cross_entropy`). The result is a real `[0, 1]` value: 1.0 means the candidate already solves the new task zero-shot, 0.0 means it is no better than the naive reference, and anything in between reflects partial, graded transfer.

The automatic plugin uses this zero-shot score as a REUSE candidate signal. When an `AdaptationCompatibilityScorer` is supplied, CLONE is evaluated separately by measuring post-adaptation improvement over a matched fresh-model control. This keeps zero-shot compatibility and adaptation value as distinct signals rather than pretending that one score answers both questions.

## Controlled arithmetic benchmark

The arithmetic benchmark compares two otherwise identical Avalanche `Naive` runs:

- **baseline**: normal sequential training with no Skill Memory plugin;
- **skill_memory**: the same model, optimizer, data, seeds, batch size, and epoch budget,
  with `SkillMemoryPlugin` enabled.

Evaluation data are independent from training data and probe data, and are never passed to the compatibility function.

For the 15-seed benchmark, the output includes final MSE, MSE history, forgetting, decisions, compatibility scores, stored skills, aggregate mean/std/95% CI, and paired per-seed differences between Skill Memory and baseline.

Forgetting is defined as `max(0, final MSE - minimum MSE observed after any experience for that task)`. The final task is excluded from forgetting because there is no later experience after which forgetting can be measured.

Because the score is now a real measurement rather than a fixed lookup, the exact decision sequence varies by seed. The script asserts structural invariants rather than one fixed sequence.

The arithmetic output is written to `results/skill_memory_benchmark_multiseed.json`.

### CLONE scorer budget

The arithmetic CLONE scorer deliberately uses a **64-sample adaptation set**, a **batch size of 16**, and **12 optimizer updates**. The 12 updates correspond to the learner's configured budget of 3 epochs over 64 samples with minibatches of 16 (`3 × 4 = 12`). The scorer therefore evaluates candidate and SCRATCH controls using the same number and size of optimizer updates instead of repeatedly applying one 16-sample probe batch.

The scorer consumes the four deterministic 16-sample batches in order and wraps to the first batch after each complete pass. This matches the **update count and minibatch structure** used by the benchmark. It does not claim to reproduce Avalanche's default randomized minibatch order; if exact sample ordering is required for a future experiment, that order must be explicitly shared between the learner and scorer as part of the benchmark configuration.

The adaptation set is separate from both the zero-shot probe and final evaluation set. It is sampled from the target experience's training distribution and is used only to estimate CLONE-versus-SCRATCH initialization value.

## Policy oracle benchmark: does automatic match the best strategy?

A positive average result (Skill Memory beats baseline) is necessary but not sufficient evidence that the automatic *policy* is good: it could still be picking the wrong strategy regularly while getting bailed out on other tasks. `examples/skill_memory_policy_oracle.py` asks the sharper, per-task question directly, using `SkillMemoryPlugin`'s `force_decision` parameter (`avalanche/training/skill_memory/plugin.py`) to fix the decision instead of deriving it from the score:

- **scratch_only**: always train from scratch;
- **clone_only**: always CLONE from the best candidate;
- **reuse_only**: always REUSE the best candidate directly;
- **automatic**: the real score-driven policy.

For every (seed, task) pair, the "oracle" strategy is whichever of scratch/clone/reuse achieved the lowest MSE, and the script reports whether `automatic` matched it.

The oracle output is written to `results/skill_memory_arithmetic_policy_oracle.json`.

## Standard Avalanche benchmark: RotatedMNIST

The next-stage benchmark uses Avalanche's standard `RotatedMNIST` scenario rather than the synthetic arithmetic stream. The transformation sequence is:

| Experience | Rotation |
| --- | ---: |
| 0 | 0° |
| 1 | 30° |
| 2 | 60° |
| 3 | 0° |
| 4 | 30° |

Repeated transformations provide legitimate reuse opportunities without hard-coding source/target task pairs. Compatibility is scored with `ProbeCompatibilityScorer`, using a small probe of the new experience's own *training* images and cross-entropy against a uniform-class-guess reference (`log(10)`); test samples and targets are never used for the reuse decision.

Four core conditions are compared:

- **naive**: standard Avalanche `Naive` training;
- **replay**: `Naive` plus Avalanche `ReplayPlugin` with a fixed replay memory;
- **skill_memory**: `Naive` plus `SkillMemoryPlugin` (automatic policy).

A second, smaller set of seeds also runs the same `force_decision`-based oracle conditions (`skill_memory_scratch_only` / `_clone_only` / `_reuse_only`) as the arithmetic policy-oracle benchmark, so the same "does automatic match the best forced strategy" question can eventually be asked on real image data too.

### Cost control (`FAST_TEST`)

This benchmark honors the `FAST_TEST` environment variable already used elsewhere in this repo's test suite. When `FAST_TEST=True`, each experience's train/eval data is subsampled to a small fixed size and only one seed runs; this exercises the same code path against real MNIST data with a CI-sized budget. Without `FAST_TEST` it runs the full dataset across `SEEDS` for a research-quality result — use this locally, not in CI.

The standard benchmark output is written to `results/skill_memory_rotated_mnist.json`.

## Interpretation

The arithmetic experiment is the controlled mechanistic test. The policy oracle experiment tests the decision mechanism itself, not just the aggregate outcome. The RotatedMNIST experiment is the first generalization test. The CLONE scorer's adaptation budget is explicitly reported so that the policy estimate can be audited against the learner's update structure.

## Running

From the repository root:

```bash
python examples/skill_memory_benchmark.py
python examples/skill_memory_policy_oracle.py
python examples/skill_memory_standard_benchmark.py       # full run
FAST_TEST=True python examples/skill_memory_standard_benchmark.py  # CI-sized run
```
