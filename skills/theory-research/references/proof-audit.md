# Proof and Result Audit

Run this audit after discovering a plausible proof, bound, or counterexample. Treat every unchecked
step as open until the required evidence exists.

## 1. Statement audit

- Are all variables introduced and typed?
- Are quantifiers explicit and ordered correctly?
- Are constants uniform or instance-dependent?
- Is asymptotic notation anchored to the correct parameter?
- Are randomness, probability space, and adversary order specified?
- Is the computational model the same in upper and lower bounds?
- Are empty, singleton, equality, boundary, and infinite cases covered?
- Does the conclusion match the original research question rather than a silently weakened one?

## 2. Dependency DAG

```markdown
| Claim ID | Formal statement | Uses | Additional assumptions | Status | Evidence location |
|---|---|---|---|---|---|
```

Topologically sort the claims. A proof step may use only earlier proved claims, definitions, and
explicitly cited theorems. If an edge needs an extra assumption, either prove it, add it to the
statement, or weaken the conclusion.

## 3. Local step audit

For each nontrivial inference check:

- implication direction and quantifier scope;
- sign, monotonicity, and inequality direction;
- division by possibly zero quantities;
- existence and uniqueness of selected objects;
- interchange of limit, derivative, expectation, sum, or integral;
- convergence mode and domination/boundedness assumptions;
- independence or conditional-independence claims;
- use of choice, compactness, measurability, regularity, or finite-dimensionality;
- preservation under a reduction, normalization, quotient, or coordinate change.

Expand “clearly,” “similarly,” and “without loss of generality.” Each must hide only a mechanical
step or a proved symmetry.

## 4. Counterexample battery

Test:

- minimum size and first non-trivial size;
- zero, one, equality, and repeated elements;
- disconnected, complete, sparse, dense, symmetric, and maximally asymmetric objects as relevant;
- deterministic and degenerate distributions;
- parameter boundaries and limiting sequences;
- adversarial instances targeting every assumption;
- removal of assumptions one at a time.

A failed stronger statement can clarify the sharp theorem. Preserve minimal counterexamples with
checker output.

## 5. Computational witness audit

Record:

```markdown
Instance definition:
Exact property checked:
Checker implementation and independent test cases:
Search method and completeness claim:
Symmetry reduction and proof of coverage:
Arithmetic mode and tolerances:
Command, environment, code revision, raw output:
Independent rerun or second checker:
```

Distinguish “found a witness” from “exhaustively proved no witness exists up to N.” Exhaustiveness
requires a coverage argument, not only a completed loop.

## 6. Bound audit

- Put upper and lower bounds in identical notation and model.
- Identify logarithmic, constant, parameter, or probability gaps.
- Trace each source of slack to a named inequality or construction restriction.
- Check whether the lower-bound hard instance satisfies all problem assumptions.
- Check whether the upper-bound algorithm is uniform, constructive, and resource-feasible as
  claimed.
- Analyze equality cases; they often reveal the sharp construction or missing assumption.

## 7. Prior-art and contribution audit

Search the final theorem statement, not the initial topic. Check synonymous definitions, equivalent
models, special cases, stronger theorems, and historical sources. Compare:

| Facet | This result | Closest result | Exact delta |
|---|---|---|---|

Facets include domain, assumptions, conclusion, bound, computational model, constructiveness, and
proof technique. Use “no equivalent result found within searched scope” rather than asserting
absolute novelty from finite retrieval.

## 8. Final result labels

Use one:

- **proved** — complete auditable proof;
- **proved conditional on cited theorem X**;
- **machine-verified** — name prover, environment, and artifact;
- **verified finite computation** — name range and completeness argument;
- **counterexample found** — independently checked;
- **partial theorem** — state exact additional assumptions or weaker conclusion;
- **conjecture supported computationally** — state searched range and method;
- **open gap** — state the first unproved dependency.

Write the contribution and limitations from this label. Elegance, confidence, or the absence of a
found counterexample cannot upgrade it.

## Research basis

Formal environments provide an executable correctness signal that natural-language proof review
does not. [AlphaProof](https://www.nature.com/articles/s41586-025-09833-y) demonstrates large-scale
proof and disproof search in Lean, while also showing that a formally valid auto-formalized statement
may fail to match the intended natural-language problem. Therefore audit both the formal proof and
the fidelity of the formal statement. Research-level performance remains limited in current public
benchmarks such as [LemmaBench](https://arxiv.org/abs/2602.24173), and false supplied statements can
still elicit convincing flawed proofs as studied in
[BrokenMath](https://arxiv.org/abs/2510.04721).
