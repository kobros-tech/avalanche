# Skill Memory benchmark

This document describes three development benchmarks for the Skill Memory integration. The arithmetic benchmark is a controlled proof-of-benefit experiment; the policy oracle benchmark asks the sharper question of whether the automatic decision is actually the *right* one; the RotatedMNIST benchmark is the first standard Avalanche scenario used to test whether the behavior extends beyond arithmetic.

## Why probe-based compatibility

Earlier versions of both the arithmetic and RotatedMNIST benchmarks scored compatibility with a hand-written binary lookup: "is this operation a declared prerequisite?" or "does the rotation match exactly?". Both can only ever return exactly `0.0` or `1.0`.

`SkillMemoryPlugin`'s three-way decision is entirely driven by where the score falls:

* `score >= reuse_threshold` -> REUSE
* `clone_threshold <= score < reuse_threshold` -> CLONE
* `score < clone_threshold` -> SCRATCH

A binary score can never land strictly between the two thresholds, so with either of the old scorers the CLONE band was **structurally unreachable** no matter how the thresholds were tuned. That is the "the transition isn't automatic" problem: CLONE was never actually exercised, so raising or lowering the thresholds could not have surfaced it either — the standard RotatedMNIST run was correctly pulled from CI rather than being trusted as evidence about CLONE.

`avalanche/training/skill_memory/compatibility.py` adds `ProbeCompatibilityScorer`, a single, continuous, generic replacement: load the candidate skill into a fresh copy of the model, measure its **zero-shot** error on a small probe drawn from the new experience's own training data (never the held-out evaluation/test stream), and compare that error against a cheap reference (a mean predictor for regression via `mean_baseline_mse`, or a uniform-class guess for classification via `uniform_guess_cross_entropy`). The result is a real `[0, 1]` value: 1.0 means the candidate already solves the new task zero-shot, 0.0 means it is no better than the naive reference, and anything in between reflects partial, graded transfer.

Both `skill_memory_benchmark.py` (arithmetic) and `skill_memory_standard_benchmark.py` (RotatedMNIST) now use `ProbeCompatibilityScorer`. Re-running the arithmetic benchmark with it shows CLONE actually firing for some seeds (e.g. seed 4: `add` clones from `multiply` at score 0.54, `square` clones from `add` at 0.32) rather than only ever SCRATCH/REUSE.

## Controlled arithmetic benchmark

The arithmetic benchmark compares two otherwise identical Avalanche `Naive` runs:

- **baseline**: normal sequential training with no Skill Memory plugin;
- **skill_memory**: the same model, optimizer, data, seeds, batch size, and epoch budget,
  with `SkillMemoryPlugin` enabled, using `ProbeCompatibilityScorer`.

Evaluation data are independent from training data and probe data, and are never passed to the compatibility function.

For the 15-seed benchmark, the output includes final MSE, MSE history, forgetting, decisions, compatibility scores, stored skills, aggregate mean/std/95% CI, and paired per-seed differences between Skill Memory and baseline.

Forgetting is defined as `max(0, final MSE - minimum MSE observed after any experience for that task)`. The final task is excluded from forgetting because there is no later experience after which forgetting can be measured.

Because the score is now a real measurement rather than a fixed lookup, the exact REUSE/CLONE/SCRATCH sequence varies by seed; the script only asserts structural invariants (first experience is always SCRATCH since memory starts empty; every decision is a valid SCRATCH/CLONE/REUSE) rather than one fixed sequence.

The arithmetic output is written to `results/skill_memory_benchmark_multiseed.json`.

## Policy oracle benchmark: does automatic match the best strategy?

A positive average result (Skill Memory beats baseline) is necessary but not sufficient evidence that the automatic *policy* is good: it could still be picking the wrong strategy regularly while getting bailed out on other tasks. `examples/skill_memory_policy_oracle.py` asks the sharper, per-task question directly, using `SkillMemoryPlugin`'s new `force_decision` parameter (`avalanche/training/skill_memory/plugin.py`) to fix the decision instead of deriving it from the score:

- **scratch_only**: always train from scratch;
- **clone_only**: always CLONE from the best candidate;
- **reuse_only**: always REUSE the best candidate directly;
- **automatic**: the real score-driven policy.

For every (seed, task) pair, the "oracle" strategy is whichever of scratch/clone/reuse achieved the lowest MSE, and the script reports whether `automatic` matched it.

**Current 15-seed result: automatic matches the oracle 93% of the time on `add` and `square`, but 0% of the time on `divide`.** Digging into `divide` specifically: cloning or reusing *any* stored skill (regardless of which one, and despite a compatibility score of `0.0` for all of them) roughly halves the MSE compared to scratch (e.g. seed 0: scratch 41.1 vs. clone/reuse 8.0). The automatic policy always chooses SCRATCH for `divide` because the zero-shot probe correctly reports that no stored skill already predicts `divide` — but zero-shot fit and initialization value are different things. A candidate can be a bad zero-shot predictor and still be a *better starting point for gradient descent* than a fresh random initialization, and the current probe only measures the former. This is a genuine limitation of probe-based compatibility as a CLONE signal, not of Skill Memory itself, and it is the clearest concrete next research question raised by this PR: the compatibility score should probably be (or be supplemented by) a measurement of post-adaptation value (e.g. error after a handful of gradient steps from that candidate) rather than zero-shot error alone, at least for deciding CLONE vs SCRATCH.

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

This benchmark was previously removed from CI for being too expensive: it trains on real MNIST images across 5 experiences x 3+ conditions x multiple seeds. Rather than dropping it again, it now honors the `FAST_TEST` environment variable already used elsewhere in this repo's test suite. When `FAST_TEST=True`, each experience's train/eval data is subsampled to a small fixed size (`FAST_TRAIN_SAMPLES` / `FAST_EVAL_SAMPLES`, 200 images by default) and only one seed runs; this exercises the exact same code path against real MNIST data, just less of it, so it should be safe and cheap to run on every CI push. Without `FAST_TEST` it runs the full dataset across `SEEDS` for a research-quality result — use this locally, not in CI.

The standard benchmark output is written to `results/skill_memory_rotated_mnist.json`.

## Interpretation

The arithmetic experiment is the controlled mechanistic test. The policy oracle experiment tests the decision mechanism itself, not just the aggregate outcome. The RotatedMNIST experiment is the first generalization test. A positive result on any one benchmark should be interpreted in context; none of them alone establishes general continual-learning superiority, and the oracle result above shows a concrete, specific gap (CLONE/REUSE-as-initialization value vs. zero-shot fit) rather than a vague "needs more work."

## Threshold calibration

`reuse_threshold=0.90` and `clone_threshold=0.30` are the values used throughout these benchmarks, unchanged from the original prototype. They have **not** been calibrated against data — with a real, continuous probe score in hand, that calibration is now possible in a way it wasn't with a binary score, but it hasn't been done yet. Recommended next step before treating these as final: sweep both thresholds against the policy-oracle script's match rate (`results/skill_memory_arithmetic_policy_oracle.json`) instead of guessing, and expect `clone_threshold` in particular to need rethinking in light of the `divide` result above — a single zero-shot-error-based score may not be able to serve both the REUSE and CLONE decisions well at once.

## Running

From the repository root:

```bash
python examples/skill_memory_benchmark.py
python examples/skill_memory_policy_oracle.py
python examples/skill_memory_standard_benchmark.py       # full run
FAST_TEST=True python examples/skill_memory_standard_benchmark.py  # CI-sized run
```

