# Theory Attack Protocol

Choose attacks from the logical form and structure of the problem. A long menu is not progress;
each branch needs a concrete subgoal, verifier, and kill condition.

## 1. Normalize the problem

Create a definition table:

```markdown
| Symbol/term | Type/domain | Definition | Depends on | Edge cases |
|---|---|---|---|---|
```

Then rewrite the target as a quantifier string and as its negation. For example:

```text
Target:  for every X in C, there exists Y such that P(X,Y).
Negation: there exists X in C such that for every Y, not P(X,Y).
```

The negation specifies the counterexample search. Identify whether the claim is monotone in any
parameter and whether compactness, finiteness, or symmetry can reduce the domain.

## 2. Select structurally distinct branches

### Construction

Use for existence or upper bounds. Start with the smallest object, compose known primitives, or
define an algorithm. Build an independent checker. Seek a generative invariant that makes every
construction step preserve feasibility.

Kill when the candidate family necessarily violates a stated obstruction.

### Minimal counterexample

Assume a counterexample minimal under size, weight, dimension, or another well-founded order. Derive
local structure forced by minimality, then reduce or transform it into a smaller counterexample.

Kill when the proposed order is not well-founded or the reduction does not preserve the class.

### Extremal or potential argument

Select an object maximizing or minimizing a meaningful quantity. Use optimality to force structure.
For iterative processes, find a potential that moves monotonically and is bounded.

Kill when equality cases or cycles leave the quantity unchanged without reaching the goal.

### Counting and probabilistic method

Count the same set in two ways, use averaging, concentration, local lemma, entropy, or random
construction. State the probability space and dependence assumptions.

Kill when the expected count or tail bound is too weak, then identify the exact slack rather than
adding opaque estimates.

### Algebraic or spectral representation

Encode combinatorial or geometric structure as a polynomial, matrix, group action, generating
function, Fourier transform, or operator. Look for rank, degree, eigenvalue, ideal, or representation
constraints.

Kill when the encoding loses the property required to translate the result back.

### Geometry, topology, and duality

Change the object into a feasible region, manifold, complex, dual program, or separating
hyperplane. Use convexity, compactness, fixed points, homology, or minimax when assumptions support
them.

Kill when regularity, convexity, closedness, orientation, or strong duality assumptions fail.

### Reduction and equivalence

Transform to a known theorem or hard problem. Prove both the mapping and preservation of instances,
solutions, parameters, and computational cost.

Kill when the reduction solves only a relaxed or differently parameterized problem.

### Induction or recurrence

Choose the induction parameter and strengthen the hypothesis enough to survive the step. Check the
base cases mechanically when possible.

Kill when the step changes an uncontrolled secondary parameter or assumes the desired structure.

## 3. Rotate representations

Try only rotations with a plausible structural payoff:

- primal ↔ dual;
- object ↔ complement;
- local constraints ↔ global invariant;
- discrete ↔ continuous relaxation;
- deterministic ↔ distribution over objects;
- static object ↔ trajectory or process;
- recursive definition ↔ generating function;
- geometry ↔ algebra or graph;
- statement ↔ contrapositive.

For each rotation, write the forward and inverse translation. If translation back is not proved, the
rotated result is a result about another problem.

## 4. Computational conjecture loop

1. Implement a property checker independently of the constructor.
2. Verify known examples and hand-built failures.
3. Enumerate or optimize the smallest non-trivial cases.
4. Store canonical objects to avoid symmetry duplicates.
5. Mine sequences, equality cases, obstructions, and invariants.
6. State a conjecture with exact domain and minimal evidence.
7. Search specifically for counterexamples beyond the discovery range.
8. Prove the structural reason or keep the result labeled computational.

Use exact arithmetic where feasible. With floating point, bound numerical error or validate the
candidate symbolically/rationally before treating it as a witness.

Executable program search can be powerful when there is an efficient, rich evaluator and a focused
program skeleton. [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) and
[AlphaEvolve](https://arxiv.org/abs/2506.13131) motivate evaluator-first search, archives, and
diversity preservation. These systems still require human-readable proof or independently verified
finite computation for theorem-level claims.

## 5. Failure ledger

```markdown
| Branch | Required lemma | Evidence tried | Failure mechanism | Definitive? | Reopening condition |
|---|---|---|---|---|---|
```

Periodically group failures. Repeated boundary terms, parity obstructions, lost convexity, or
noncommuting operations may reveal the real theorem.

## 6. Partial-result ladder

When the full statement resists attack, seek in order of scientific value:

- a counterexample to an overstrong formulation;
- a sharp special case;
- proof under one explicit additional assumption;
- improved bound with identified remaining slack;
- equivalence to a named subproblem;
- structural lemma that eliminates a broad attack family;
- a verified finite range with a proof reducing all remaining cases to a clear obstacle.

Do not relabel a convenient special case as completion of the original claim.
