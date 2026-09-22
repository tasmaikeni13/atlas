---
name: theory-research
description: Attack mathematical and theoretical-computer-science research problems with formal conjectures, counterexample search, proofs, bounds, and machine-checked or computational witnesses where possible. Use when asked to prove or disprove a claim, find tight bounds, solve an open-form theorem problem, construct an extremal object, or establish theory for a new algorithm.
---

# Theory Research

Turn a mathematical question into one of four auditable outcomes: a proof, a counterexample, a
verified construction, or a sharply bounded partial result. Plausible prose and numerical pattern
matching are not proofs.

## 1. Freeze the statement before attacking it

Write the exact objects, domains, quantifiers, assumptions, model of computation, and target
conclusion. Translate the statement back into plain language and test degenerate cases. Distinguish:

- existence from construction;
- pointwise from uniform claims;
- asymptotic from finite claims;
- worst-case from average-case or high-probability claims;
- upper from lower bounds and the model in which each is measured.

If the supplied statement is ambiguous, enumerate the materially different interpretations and
choose only after their consequences are visible.

## 2. Establish status and nearby results

Search current primary literature for the exact statement, equivalent formulations, special cases,
known barriers, and strongest bounds. Trace definitions carefully; the same term may encode a
different model. Report whether the problem is known, open within searched scope, or a variation of
known work. Never infer “open” from failure to remember a theorem.

## 3. Build the verifier before the witness

Whenever the objects are finite or computable, implement an independent property checker first.
Test it on known positive and negative examples. Then enumerate, optimize, use SAT/SMT, symbolic
algebra, theorem provers, or numerical search as appropriate.

Computation may:

- falsify a universal conjecture;
- discover minimal examples and invariants;
- verify a finite residual case;
- suggest a lemma or sharp constant.

It does not prove an unbounded claim unless the reduction to the checked finite cases is itself
proved.

## 4. Maintain distinct attack branches

Use structurally different branches, not cosmetic proof styles. Relevant families include:

- direct construction or algorithm;
- minimal counterexample and obstruction;
- extremal or potential-function argument;
- duality, relaxation, or separation;
- probabilistic method or counting;
- algebraic, spectral, geometric, or topological representation;
- reduction to or from a known result;
- induction, recurrence, compactness, or limiting argument.

For each branch record the key lemma it needs, why that lemma might hold, and a condition that kills
the branch. Read [references/attack-protocol.md](references/attack-protocol.md) for routing and
computational-search guidance.

## 5. Alternate construction and obstruction

Do not spend the entire search trying to prove the preferred statement. Repeatedly attempt to break
it:

- enumerate the smallest non-trivial cases;
- push parameters to zero, equality, boundary, and infinity;
- remove each assumption;
- search adversarially against the proposed invariant;
- ask whether a stronger claim is false and a weaker one is sufficient;
- inspect where every failed proof uses an unstated condition.

Classify failures by mechanism. “The approach did not work” is not reusable knowledge; “the
induction loses monotonicity when the boundary term changes sign” is.

## 6. Promote observations through an evidence ladder

Keep labels strict:

1. observed numerically;
2. conjectured with explicit domain;
3. proved under additional assumptions;
4. proved as stated;
5. independently or machine verified.

Never skip a level in the final presentation. If a numerical pattern guides the proof, retain the
search script and examples but make the proof logically independent of floating-point coincidence.

## 7. Close bounds and dependencies

For a bound, seek the matching construction or obstruction in the same model. Identify every source
of slack. For a proof, build a dependency DAG from definitions through lemmas to the main theorem.
Each edge must use only stated hypotheses. If a lemma is stronger than needed, weaken it; if it is
false, preserve the counterexample and revise the main claim.

## 8. Audit the result adversarially

Read [references/proof-audit.md](references/proof-audit.md) before presenting a research-grade result.
At minimum:

- verify quantifiers and variable scope;
- check all divisions, limits, convergence, measurability, and existence assumptions;
- test equality and degenerate cases;
- distinguish cited lemmas from newly proved steps;
- rerun computational witnesses independently;
- compare the final statement, not the initial intuition, with closest prior work;
- state exactly what remains unproved.

## Required deliverable

Provide the formal statement, status against literature, result label, proof or counterexample,
dependency structure, computational artifacts and commands when used, assumption audit, and the
remaining gap. If incomplete, leave a precise frontier: strongest proved lemma, smallest known
counterexample-free range, failed branches with mechanisms, and the next decisive subproblem.
