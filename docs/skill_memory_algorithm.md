# Skill Memory Algorithm

## Status

This document is the normative definition of the Skill Memory algorithm implemented and investigated in this branch. It takes precedence over prototype wording that treats REUSE and CLONE as equivalent model initialization operations.

## 1. Objective

Skill Memory is a continual skill-acquisition mechanism. Its purpose is to let a learner acquire new skills sequentially while preserving previously acquired skills and exploiting useful prior skills when a later experience is compatible.

The central invariant is:

> **A learned skill is never modified by training for a later skill.**

A skill may be **reused** without modification, or **cloned** into a new reserved region and the clone may then be modified by training. These are different operations.

Skill Memory is not defined as replay, and it is not merely a checkpoint registry.

## 2. Terminology

### Skill

A learned capability represented by a model parameter region (a reserved skill slot) together with the metadata needed to identify and score it.

### Reserved skill slot

A region of model parameters/neuron capacity assigned to exactly one acquired skill. Once a slot has been committed to a learned skill, its parameters are protected from subsequent training unless that skill is explicitly being trained as a new clone in a different slot.

Slots are allocated in deterministic order: slot 0, slot 1, slot 2, and so on. The algorithm therefore grows the collection of acquired skills through successive reserved regions rather than rewriting an earlier skill.

### REUSE

Use an existing skill directly. The selected skill's reserved parameters remain frozen. No gradient update is applied to that skill and no replacement copy is created. REUSE therefore cannot change the learned source skill.

### CLONE

Copy the selected source skill from its reserved slot into the next unused reserved slot. The destination slot becomes the active skill for the new experience. Training is allowed to modify only the destination slot (and any explicitly designated shared parameters, if the model architecture declares them). The source slot remains unchanged.

### SCRATCH

Allocate the next unused reserved slot and initialize it from the normal fresh initialization. Train the new skill in that slot.

## 3. Sequential algorithm

For experiences `E_1, E_2, ..., E_T`:

1. Start with an empty Skill Memory and the first unused reserved slot.
2. Receive the next experience through the normal Avalanche experience lifecycle.
3. Compute a compatibility score for each stored skill using only information permitted for training-time decision making. Never use held-out test labels.
4. Select the highest-scoring candidate.
5. Apply the decision policy:

```text
if memory is empty:
    SCRATCH
elif score >= reuse_threshold:
    REUSE
elif score >= clone_threshold:
    CLONE
else:
    SCRATCH
```

Default experimental thresholds are `reuse_threshold = 0.90` and `clone_threshold = 0.30`. They are configurable hypotheses, not fixed scientific constants.

6. For **REUSE**, route the experience to the selected skill slot and protect that slot from optimizer updates. The existing skill remains exactly as stored.
7. For **CLONE**, copy the selected slot into the next unused slot, activate the new slot, and train the new slot. The source slot is protected.
8. For **SCRATCH**, initialize the next unused slot normally and train it.
9. After a successful CLONE or SCRATCH acquisition, register the destination slot as a new skill in memory.
10. After REUSE, do not replace or update the source skill record merely because the experience was processed.
11. Continue to the next experience.

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
             +-----------------+-----------------+
             |                 |                 |
        high score        medium score      low/no score
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

## 8. Compatibility score

Compatibility is a policy input, not the definition of a skill. The scorer should estimate how useful a stored skill is for the new experience using independent training/calibration information.

A continuous score in `[0, 1]` is preferred for automatic policy experiments because it permits all three regimes. A binary prerequisite lookup is useful as an oracle/control but cannot exercise a continuous threshold band.

For a candidate with probe error `L_skill` and reference error `L_ref`, one possible normalized score is:

```text
score = clamp(1 - L_skill / L_ref, 0, 1)
```

The probe must come from allowed training/calibration data, never from the final evaluation stream.

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

## 11. Evaluation

The primary scientific comparison is:

1. **Scratch:** each target skill starts in a fresh slot.
2. **Clone:** the target starts from the selected prior skill copied into a new slot.
3. **Automatic Skill Memory:** the score selects REUSE, CLONE, or SCRATCH.

All three must use the same model capacity accounting, training budget, seeds, and final evaluation data.

Measure:

- final target performance;
- training needed to reach a fixed target performance;
- decision sequence;
- selected source skill;
- compatibility score;
- benefit of the decision;
- runtime overhead;
- memory/parameter overhead;
- preservation of every previously learned skill.

The arithmetic sequence should contain genuine dependencies, such as learning multiplication before square, so that CLONE tests useful transfer rather than an arbitrary checkpoint copy.

## 12. Avalanche integration boundary

Avalanche should provide the normal sequential experience lifecycle, optimizer, model, datasets, evaluation, and plugin mechanism.

Skill Memory should provide:

- skill registry;
- compatibility lookup;
- acquisition decision;
- slot allocation/state isolation;
- integration hooks for activating/freeze/trainable regions.

The implementation should remain small and model-agnostic at the registry/policy level. Model-specific neuron partitioning belongs behind an explicit adapter rather than being guessed by the generic registry.

## 13. Minimal reference pseudocode

```python
for experience in stream:
    candidate, score = memory.best_match(experience)

    if candidate is None or score < clone_threshold:
        decision = SCRATCH
        destination = slots.next_free()
        slots.initialize_fresh(destination)
        slots.activate_trainable(destination)

    elif score < reuse_threshold:
        decision = CLONE
        destination = slots.next_free()
        slots.copy(candidate.slot, destination)
        slots.freeze(candidate.slot)
        slots.activate_trainable(destination)

    else:
        decision = REUSE
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

## 14. Definition of done

An implementation is faithful to this algorithm only when all of the following are true:

- [ ] Previously acquired skill parameters are protected from later skill training.
- [ ] REUSE does not update the reused skill.
- [ ] CLONE creates a physically independent destination parameter region.
- [ ] Training a clone changes the clone and not its source.
- [ ] New skills consume reserved slots monotonically.
- [ ] SCRATCH creates a new slot rather than overwriting an old skill.
- [ ] Only CLONE and SCRATCH create new learned-skill records.
- [ ] The automatic policy is score-driven and configurable.
- [ ] No test/evaluation labels enter compatibility decisions.
- [ ] The arithmetic proof of concept demonstrates source preservation and clone adaptation.
- [ ] Avalanche's normal sequential training lifecycle remains the integration mechanism.
