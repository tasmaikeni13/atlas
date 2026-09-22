# Method Search for ML Research

Use these axes to generate structurally distinct, testable methods. Start from the target failure
and evaluator. Do not enumerate every cell or combine components without a mechanism.

## 1. Decompose the existing method

Represent the baseline as:

```text
data -> representation -> interactions/state -> prediction/action
     -> objective/feedback -> credit assignment -> parameter update
```

For each edge record what information passes, its scale and timing, and the assumption it encodes.
Most useful research changes one load-bearing edge and predicts an observable consequence.

## 2. Architecture axes

### Information topology

- local, global, hierarchical, recurrent, sparse, routed, or message-passing interactions;
- directionality and update order;
- shared versus private state;
- fixed versus input-conditioned connectivity.

Ask which dependency the baseline cannot express or expresses wastefully. Match parameter count and
active compute when testing topology itself.

### State and memory

- stateless, finite-window, recurrent, external memory, multiscale, or continuous state;
- write, read, forget, and reset rules;
- observability and information lifetime.

Derive a task where the required dependency length or sufficient statistic is known.

### Symmetry and coordinates

- invariance, equivariance, canonicalization, quotienting, coordinate-free operations;
- local versus global frames;
- learned versus imposed symmetry.

Test transformations that should preserve or predictably transform outputs. An imposed false
symmetry can reduce performance.

### Capacity allocation

- width/depth, conditional computation, experts, adaptive depth, low-rank state, sparsity;
- static versus uncertainty- or difficulty-dependent allocation.

Control total and active capacity. Measure routing collapse, utilization, and overhead.

### Discrete versus continuous computation

- layer stacks, recurrent steps, differential equations, event-driven updates, iterative solvers;
- stable discretization and tolerance.

Check whether gains arise from more function evaluations rather than the proposed dynamics.

## 3. Optimizer axes

### Geometry and preconditioning

- Euclidean, mirror, natural-gradient, trust-region, proximal, or manifold updates;
- diagonal, block, low-rank, Kronecker, or implicit curvature estimates.

Prediction targets: coordinate sensitivity, conditioning, curvature alignment, or constraint
preservation. Cost targets: extra memory, matrix operations, and numerical damping.

### Temporal state

- momentum, averaging, adaptive moments, gradient differences, learned state, multiscale memory;
- bias correction, reset, decay, and timescale.

Test delayed, noisy, rotating, and nonstationary gradients. Separate state benefit from a changed
effective learning-rate schedule.

### Noise and sampling

- minibatch selection, variance reduction, perturbation, Langevin noise, antithetic samples,
  control variates;
- adaptive batch or fidelity.

State whether noise is an estimator defect, exploration mechanism, or regularizer. Each role implies
different diagnostics.

### Constraint handling

- projection, penalty, barrier, reparameterization, dual variables, clipping, normalization;
- hard versus soft feasibility and delayed constraint correction.

Measure violation and objective jointly. A lower loss outside the feasible set is not progress.

### Update scheduling

- synchronous/asynchronous, layerwise, event-triggered, alternating, optimistic, or
  extragradient updates;
- learning-rate, weight-decay, and state schedules.

Control update count and stale information. Derive a regime in which scheduling changes should
matter.

## 4. Objective and feedback axes

- target quantity versus surrogate;
- decomposition across examples, tokens, steps, views, or constraints;
- static versus adaptive weighting;
- robust, risk-sensitive, calibrated, contrastive, generative, or self-supervised feedback;
- local versus global credit assignment;
- curriculum, active sampling, or adversarial example generation.

Ask what estimator the objective implements and under what data-generating assumptions. A new loss
needs a proper target, not only a plausible shape.

## 5. Mutation operators for hypotheses

Apply selectively:

- remove a component and restore only the capability its absence exposes;
- replace an approximate operation with a principled estimator or solver;
- turn a fixed quantity into state-dependent control with an explicit observation;
- separate coupled timescales or couple variables whose interaction is currently ignored;
- move computation from training to inference or the reverse;
- change coordinates or dualize a constraint;
- expose a hidden state and design an observer;
- convert a hard selection to a continuous relaxation, then test relaxation error;
- import a source mechanism only after writing its relational map and boundary.

Every mutation must produce a hypothesis card and a unique prediction.

## 6. Portfolio rules

- Deduplicate candidates by causal delta and load-bearing assumption.
- Preserve lineage: parent, mutation, expected change, observed result.
- Keep a few candidates in different regions of the design space even if their initial score is
  lower.
- Do not combine two candidates until each has an independently supported sub-mechanism or the
  combination makes a distinct synergy prediction.
- Re-open a killed branch only when the failure mechanism or test regime changes.

Program-search systems provide useful operational lessons when a rich automated evaluator exists:
[FunSearch](https://www.nature.com/articles/s41586-023-06924-6),
[AlphaEvolve](https://arxiv.org/abs/2506.13131), and
[Empirical Research Assistance](https://www.nature.com/articles/s41586-026-10658-6). They motivate
executable candidates, population memory, evaluator-first design, and explicit exploration versus
exploitation. They do not justify optimizing a proxy that is scientifically invalid.

Idea review is also an unreliable substitute for execution. A large human study found weaknesses
in LLM self-evaluation and diversity, and a follow-up execution study found that apparent ideation
advantages shrank or reversed after researchers implemented the proposals:
[Can LLMs Generate Novel Research Ideas?](https://arxiv.org/abs/2409.04109) and
[The Ideation–Execution Gap](https://arxiv.org/abs/2506.20803). This is why the portfolio moves
quickly to discriminating code and experiments instead of repeatedly polishing proposal prose.
