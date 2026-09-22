---
name: mechanism-transfer
description: Search for and translate mechanisms from other disciplines into testable target-domain designs. Use when borrowing mathematics, physics, biology, control, optimization, economics, or another field to create an architecture, optimizer, algorithm, system, or scientific hypothesis; requires a relational mapping, a concrete reconstruction, and a kill test rather than a surface analogy.
---

# Mechanism Transfer

Turn a source-domain mechanism into a target-domain object that can be derived, implemented, and
falsified. A transfer succeeds only if the source’s causal or mathematical structure survives the
mapping and produces a target prediction that the unmodified baseline does not.

## Output contract

The finished work must include:

- a precise target failure or missing capability;
- evidence-backed source mechanism cards;
- an explicit source-to-target relation map;
- a domain-neutral bridge model;
- a target reconstruction in equations, pseudocode, code, or experimental protocol;
- a compatibility audit and a cheapest kill test;
- a scoped novelty result after the design is concrete.

If the task permits implementation or computation, do not stop at ideation. Build the smallest
testable version and run the cheapest discriminating check.

## 1. Specify the target socket

State what the imported mechanism must do:

- observed failure, bottleneck, or unexplained behavior;
- target variables, interfaces, and controllable operations;
- invariants and constraints that cannot be violated;
- evaluator, metric, or proposition that can reject the design;
- resource envelope and acceptable tradeoffs.

Translate broad aims such as “improve optimization” into an observable signature such as reduced
oscillation under delayed gradients, scale-invariant steps, or lower regret under a stated regime.
Without a rejectable signature, cross-domain search will produce decoration.

## 2. Derive a mechanism request

Work backward from the target signature. Describe the missing job in relational language, without
naming a preferred source domain. For example: “adapt a local update using a bounded memory of
curvature while remaining stable when observations are noisy and delayed.”

List the minimum relations that any valid mechanism must instantiate. This request becomes the
retrieval key and prevents fixation on a fashionable analogy.

## 3. Search for structurally diverse sources

Read [references/search-and-mapping.md](references/search-and-mapping.md) before a serious transfer.
Search by relational verbs, invariants, failure signatures, and equations—not just target nouns.
Pursue both:

- near sources with compatible mathematics and low translation risk;
- distant sources with a different implementation of the same relational job.

Use multiple source examples when possible. Comparing two implementations of the same mechanism
helps separate the invariant from incidental domain features. Preserve a portfolio of structurally
different candidates; do not collapse everything into the first plausible source.

Use [references/mechanism-families.md](references/mechanism-families.md) as search vocabulary, not as
an idea menu. A named family is not evidence that its assumptions hold in the target.

## 4. Build source mechanism cards

For every serious source, extract:

- state variables and their types;
- governing relations, operations, or causal chain;
- the behavior produced and evidence for that link;
- invariants, boundary conditions, and timescale;
- required observations and control channels;
- failure regimes and counterexamples;
- primary source and confidence.

Apply a removal or intervention test: if deleting the proposed mechanism would not remove the source
behavior, the extraction is probably a correlate or story rather than a mechanism.

## 5. Construct the relational map and bridge model

Map relations before objects. Preserve causal order, higher-order relations, constraints, and
directionality. For each mapped element, state:

| Source role | Source relation | Target role | Target relation | Mismatch or adaptation |
|---|---|---|---|---|

Then write a domain-neutral bridge model containing only typed variables, transformations,
dynamics, invariants, and boundary conditions. The bridge must be specific enough to simulate or
derive consequences. “Both systems self-organize” is not a bridge model.

Reject the transfer if types, units, scale, observability, causality, or required interventions do
not match. When a mismatch is repairable, name the adapter and the new assumption it introduces.

## 6. Reconstruct the target mechanism

Implement the bridge using target-native objects. Define:

- changed components and unchanged baseline;
- update equations, data flow, or intervention protocol;
- added state, parameters, compute, and latency;
- initialization and limiting behavior;
- what source properties are intentionally not transferred.

Prefer the smallest causal delta. If five ideas change at once, a positive result cannot identify
the responsible mechanism.

## 7. Derive risky predictions and kill the idea cheaply

Before full evaluation, state at least one prediction that is unusually likely if the transfer is
real and unlikely under a generic capacity or tuning explanation. Test with:

- mechanism removal or corruption;
- a regime where the source theory predicts the effect should vanish or reverse;
- a matched-compute and matched-parameter baseline;
- a scale, delay, noise, or boundary sweep;
- a toy system where the proposed invariant can be measured directly.

If the idea survives, progress to stronger evaluation. If it fails, record whether the source
mechanism was wrong, the map was invalid, the adapter broke an invariant, or the target simply did
not need the mechanism. A precise negative transfer is useful evidence.

## 8. Check novelty only after reconstruction

Search the target equations, algorithmic components, compositions, and predicted behavior using
synonyms and adjacent terminology. Compare the closest work by mechanism, not branding. Report
“no close overlap found within searched scope” when that is the evidence; never promote a search
failure to a novelty proof.

## Quality gate

A transfer is ready for a research claim only when:

1. the source mechanism is supported by primary evidence;
2. the relational map is explicit and its mismatches are addressed;
3. the target reconstruction is executable or formally specified;
4. at least one risky prediction and kill test are defined;
5. simple confounds such as extra parameters, compute, or tuning are controlled;
6. the claimed scope matches the tested regime;
7. the closest prior implementation has been sought and compared.
