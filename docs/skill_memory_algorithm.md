# Skill Memory Algorithm

## Status

This document is the normative definition of the Skill Memory algorithm investigated in this branch. It takes precedence over prototype wording that treats REUSE and CLONE as equivalent model initialization operations.

**The algorithm does not prescribe fixed compatibility-score values or fixed score thresholds.** The compatibility score is evidence used by an acquisition policy. Score calibration and policy selection are experimental components and must be evaluated separately from the underlying skill-memory mechanism.

## 1. Objective

Skill Memory is a continual skill-acquisition mechanism. Its purpose is to let a learner acquire new skills sequentially while preserving previously acquired skills and exploiting useful prior skills when a later experience is compatible.

The central invariant is:

> **A learned skill is never modified by training for a later skill.**

A skill may be **reused** without modification, or **cloned** into a new reserved region and the clone may then be modified by training. These are different operations.

Skill Memory is not defined as replay, and it is not merely a checkpoint registry.

## 2. Terminology

### Skill

A learned capability represented by a model parameter/neuron region (a reserved skill slot) together with the metadata needed to identify and score it.

### Reserved skill slot

A region of model parameters/neuron capacity assigned to exactly one acquired skill. Once a slot has been committed to a learned skill, its parameters are protected from subsequent training.

Slots are allocated in deterministic order: slot 0, slot 1, slot 2, and so on. The algorithm therefore grows the collection of acquired skills through successive reserved regions rather than rewriting an earlier skill.

### REUSE

Use an existing skill directly. The selected skill's reserved parameters remain frozen. No gradient update is applied to that skill and no replacement copy is created.

REUSE is a **use operation, not an acquisition operation**.

### CLONE

Copy the selected source skill from its reserved slot into the next unused reserved slot. The destination slot becomes the active skill for the new experience. Training is allowed to modify only the destination slot (and any explicitly designated shared parameters, if the model architecture declares them). The source slot remains unchanged.

CLONE is an **acquisition operation**.

### SCRATCH

Allocate the next unused reserved slot and initialize it from the normal fresh initialization. Train the new skill in that slot.

SCRATCH is an **acquisition operation**.

## 3. Sequential algorithm

For experiences `E_1, E_2, ..., E_T`:

1. Start with an empty Skill Memory and the first unused reserved slot.
2. Receive the next experience through the normal Avalanche experience lifecycle.
3. Compute a compatibility score for each stored skill using only information permitted for training-time decision making. Never use held-out final evaluation/test labels.
4. Select the highest-scoring candidate, if one exists.
5. Apply a **predefined acquisition policy** to the candidate and its score. The policy decides among REUSE, CLONE, and SCRATCH.
6. For **REUSE**, activate the selected existing skill and keep its reserved parameters frozen. Do not train it and do not create a new skill record.
7. For **CLONE**, copy the selected source slot into the next unused slot, activate the destination slot, and train only the destination. The source remains protected.
8. For **SCRATCH**, initialize the next unused slot normally and train it.
9. After a successful CLONE or SCRATCH acquisition, register the destination slot as a new skill in memory.
10. Continue to the next experience.

### Acquisition policy and score calibration

The algorithm deliberately does **not** define universal numerical boundaries such as `0.90` for REUSE or `0.30` for CLONE.

The compatibility score answers:

> **How useful does the available evidence suggest this stored skill is for the new experience?**

The acquisition policy answers a different question:

> **Given that evidence, should we reuse the skill, clone it and adapt it, or learn from scratch?**

These should not be conflated.

A threshold policy is allowed as one experimental implementation, but its thresholds must be calibrated using data independent of the final evaluation set. The final algorithm specification does not require thresholds at all.

A stronger calibration procedure is to evaluate candidate skills on independent calibration tasks and directly compare the three possible actions:

- **REUSE:** use the candidate unchanged and measure whether it already satisfies the target criterion;
- **CLONE:** initialize a new reserved slot from the candidate and measure its learning efficiency;
- **SCRATCH:** initialize a fresh reserved slot and measure its learning efficiency.

This calibration establishes which observable score/evidence patterns correspond to REUSE, CLONE, or SCRATCH. The resulting policy must be fixed before final test evaluation.

## 4. State-transition model

```text
                         new experience
                               |
                               v
                       score all skills
                               |
                               v
                       best candidate
                               |
                               v
                    predefined policy
                               |
             +-----------------+-----------------+
             |                 |                 |
           REUSE             CLONE            SCRATCH
             |                 |                 |
             v                 v                 v
       existing slot    copy source slot   fresh new slot
       stays frozen     -> next slot       -> next slot
             |                 |                 |
             |                 v                 v
             |              train clone      train new skill
             |                 |                 |
             |                 +--------+--------+
             |                          |
             |                          v
             |                   register new slot
             |                          |
             +--------------------------+
```

## 5. The critical REUSE/CLONE distinction

The following behavior is **forbidden**:

```text
Skill A -> load Skill A -> train new task -> Skill A changes
```

That is not REUSE. It destroys the immutability guarantee.

The required behavior is:

```text
Skill A (slot 0)
       |
       +---- REUSE ----> use slot 0, do not update slot 0
       |
       +---- CLONE ----> copy slot 0 -> slot 1 -> train slot 1
```

After CLONE:

```text
slot 0 = original Skill A       [protected]
slot 1 = adapted Skill A/B      [trainable]
```

The adapted skill is a distinct acquired skill and must have its own memory record and slot identity.

## 6. Why reserved slots are part of the algorithm

A whole-model checkpoint copy is not sufficient to express the intended algorithm. A whole-model copy can provide an initialization, but it does not establish which neurons belong to which skill or protect those neurons during later training.

The implementation therefore needs a model/strategy adapter that knows how a model is partitioned into reserved skill regions. The generic Skill Memory layer stores and indexes skills; the adapter owns the actual parameter/neuron routing and protection.

The first implementation may use a simple explicit slot architecture for the arithmetic proof of concept. It must not silently pretend that arbitrary unpartitioned Avalanche models already provide isolated neuron slots.

## 7. Skill-memory invariants

1. **Immutability:** a registered source skill never changes as a side effect of another skill's training.
2. **Unique ownership:** one reserved slot represents one acquired skill version.
3. **Monotonic allocation:** once a slot has been assigned, the next newly acquired skill uses the next unused slot; an older slot is not overwritten.
4. **Clone independence:** modifying a clone must not modify its source.
5. **Reuse isolation:** REUSE must not create a new trainable version of the selected skill.
6. **Explicit acquisition:** only CLONE and SCRATCH create new skills.
7. **No test leakage:** compatibility and policy decisions must not inspect final evaluation/test labels.
8. **Sequential processing:** decisions are made as experiences arrive, not after seeing the complete future stream.
9. **Deterministic slot identity:** slot allocation is reproducible for a fixed experience sequence.
10. **Capacity accounting:** every newly acquired skill consumes a distinct reserved slot; experiments must report the resulting capacity/parameter cost.

## 8. Compatibility score

Compatibility is a policy input, not the definition of a skill. The scorer should estimate how useful a stored skill is for the new experience using independent training/calibration information.

A continuous score is useful for experiments, but the algorithm does not require a particular numeric scale. The important requirement is that the score have a defined interpretation and be computed without final test/evaluation labels.

A binary prerequisite lookup can be used as an oracle/control, but it is not itself evidence that an automatic compatibility scorer works.

If a normalized error-based score is used, one possible definition is:

```text
score = clamp(1 - L_skill / L_ref, 0, 1)
```

This is an example, not a mandated formula. The exact scoring function must be reported with the experiment.

## 9. Training semantics

The active optimizer must update only parameters belonging to the active trainable skill region, plus any parameters explicitly declared shared by the model adapter.

For REUSE, the selected skill region is frozen. The implementation must not rely solely on loading a copied state dictionary and then continuing ordinary training, because ordinary training would turn that operation into adaptation and therefore violate REUSE semantics.

For CLONE, the source region is frozen and the destination region is trainable. The destination starts as an exact copy of the source region before training.

For SCRATCH, the destination region receives fresh initialization.

## 10. Registration semantics

A new memory record is created only after CLONE or SCRATCH completes acquisition. Its metadata should include at least:

- skill name/identifier;
- slot identifier;
- acquisition mode (`clone` or `scratch`);
- source skill identifier, if cloned;
- compatibility score used for the decision;
- experience/task metadata;
- creation order.

A REUSE event may be logged for evaluation, but it must not replace the stored source state or create an adapted record.

## 11. Scientific success criteria and evaluation

The algorithm should not be judged by whether a compatibility score exceeds an arbitrary number. The score is only a mechanism for making an acquisition decision.

The primary scientific target is:

> **Improve target-task learning relative to a scratch baseline when prior skills are useful, while preserving previously learned skills.**

### 11.1 Scratch baseline

For every target experience, train the target from a fresh reserved slot using the same architecture, capacity budget, training budget, optimizer, data, and comparable random seeds.

### 11.2 Transfer benefit

For a target with a genuinely useful prior skill, CLONE is successful when it provides measurable benefit over SCRATCH. Benefit should be measured primarily by:

- lower final target error under a fixed training budget; and/or
- fewer training steps/epochs to reach the same target criterion.

A result should be reported across multiple seeds with uncertainty, not from a single run.

### 11.3 REUSE validity and benefit

REUSE is successful only when using the existing skill unchanged is sufficient for the new task. Evidence should include:

- target performance is adequate without updating the skill; and
- the reused skill remains unchanged.

If adaptation is required, the correct acquisition operation is CLONE rather than REUSE.

### 11.4 Preservation

After acquiring each new skill, evaluate previously acquired skills again. Previously learned skills should not materially degrade relative to their performance immediately after acquisition.

The strongest mechanical test is parameter/state isolation: a source slot must remain bitwise unchanged when its clone is trained, subject only to explicitly documented shared parameters.

### 11.5 Automatic-policy success

The automatic Skill Memory policy is successful when, across multiple seeds and target tasks, it provides statistically supported improvement over the scratch baseline on transfer-appropriate tasks **without causing unacceptable degradation on non-transfer tasks or previously learned skills**.

Report:

- mean and dispersion/confidence intervals across seeds;
- final target performance;
- training steps to target performance;
- decision distribution (REUSE/CLONE/SCRATCH);
- selected source skills;
- compatibility scores;
- preservation of prior skills;
- parameter/capacity overhead;
- runtime overhead.

The policy and any score calibration procedure must be fixed before final test evaluation.

## 12. Experimental hierarchy

Experiments should be performed in the following order so that policy errors are not confused with mechanism errors.

### Level 1 — Mechanism validity

Prove with unit tests that:

```text
REUSE   -> source unchanged
CLONE   -> source unchanged + clone changes
SCRATCH -> new slot independent of all previous slots
```

### Level 2 — Oracle transfer

Provide the correct acquisition decision externally. Demonstrate that when CLONE is appropriate, a cloned skill can learn the target faster and/or better than SCRATCH.

This isolates the value of the cloning mechanism from the compatibility scorer.

### Level 3 — Automatic policy

Use the compatibility score to select REUSE/CLONE/SCRATCH using a policy fixed before final testing. Evaluate the policy against SCRATCH and appropriate control policies on unseen target tasks.

This isolates the quality of the decision mechanism.

## 13. Avalanche integration boundary

Avalanche should provide the normal sequential experience lifecycle, optimizer, model, datasets, evaluation, and plugin mechanism.

Skill Memory should provide:

- skill registry;
- compatibility lookup;
- acquisition decision;
- slot allocation/state isolation;
- integration hooks for activating, freezing, and training reserved regions.

The implementation should remain small and model-agnostic at the registry/policy level. Model-specific neuron partitioning belongs behind an explicit adapter rather than being guessed by the generic registry.

## 14. Minimal reference pseudocode

```python
for experience in stream:
    candidate, score = memory.best_match(experience)
    decision = policy(candidate, score, calibration)

    if candidate is None or decision == SCRATCH:
        destination = slots.next_free()
        slots.initialize_fresh(destination)
        slots.activate_trainable(destination)

    elif decision == CLONE:
        destination = slots.next_free()
        slots.copy(candidate.slot, destination)
        slots.freeze(candidate.slot)
        slots.activate_trainable(destination)

    elif decision == REUSE:
        destination = candidate.slot
        slots.freeze(destination)
        slots.activate(destination)

    train_current_experience_only_if(decision != REUSE)

    if decision in (CLONE, SCRATCH):
        memory.register(
            slot=destination,
            source=candidate if decision == CLONE else None,
            metadata=experience.metadata,
        )
```

The exact mechanism used to route data to a slot may differ between architectures, but the semantics above must remain unchanged.

## 15. Definition of done

An implementation is faithful to this algorithm only when all of the following are true:

- [ ] Previously acquired skill parameters are protected from later skill training.
- [ ] REUSE does not update the reused skill.
- [ ] CLONE creates a physically independent destination parameter region.
- [ ] Training a clone changes the clone and not its source.
- [ ] New skills consume reserved slots monotonically.
- [ ] SCRATCH creates a new slot rather than overwriting an old skill.
- [ ] Only CLONE and SCRATCH create new learned-skill records.
- [ ] The compatibility score has a documented interpretation.
- [ ] The acquisition policy is specified independently from the score definition.
- [ ] Score/policy calibration is performed without final test/evaluation labels.
- [ ] No arbitrary score thresholds are treated as universal algorithm constants.
- [ ] The arithmetic proof of concept demonstrates source preservation and clone adaptation.
- [ ] Oracle transfer demonstrates measurable improvement over scratch where transfer is expected.
- [ ] The automatic policy is evaluated against scratch across multiple seeds.
- [ ] Previous skills are re-evaluated after subsequent acquisitions.
- [ ] Capacity and runtime costs are reported.
- [ ] Avalanche's normal sequential training lifecycle remains the integration mechanism.
