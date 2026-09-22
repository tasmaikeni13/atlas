# Mechanism Families as Retrieval Vocabulary

Use this catalog to generate source-search terms and compatibility questions. Do not select a
family merely because it sounds sophisticated. Every transfer still needs primary evidence,
relational mapping, target reconstruction, and a kill test.

## Control and dynamical systems

### Feedback, integral action, and observers

- Search for: negative/positive feedback, delayed feedback, state observer, Kalman filtering,
  integral control, model predictive control, adaptive control.
- Target symptom: persistent error, oscillation, partial observation, changing dynamics.
- Transported structure: a state transition, measurement channel, controller, and closed-loop
  stability relation.
- Check: observability, controllability, delay, gain, noise model, and whether the controller can
  actually intervene on the target.

### Lyapunov, passivity, and contraction

- Search for: Lyapunov function, dissipativity, passivity, contraction metric, input-to-state
  stability.
- Target symptom: exploding trajectories, unstable training, sensitivity to perturbation.
- Transported structure: a scalar or metric whose evolution certifies boundedness or convergence.
- Check: the candidate quantity truly decreases or remains bounded under the target update; visual
  smoothness is not a stability proof.

### Bifurcation, hysteresis, and multiscale dynamics

- Search for: phase portrait, bifurcation, metastability, hysteresis, singular perturbation,
  timescale separation.
- Target symptom: abrupt regime changes, path dependence, fast/slow state interactions.
- Check: the target has the control parameter and separated timescales assumed by the source model.

## Geometry and mechanics

### Metric-aware and constrained motion

- Search for: Riemannian gradient, natural gradient, mirror descent, geodesic flow, projected or
  constrained dynamics, symplectic integration.
- Target symptom: poor conditioning, parameterization dependence, drift from constraints.
- Transported structure: state space, metric or divergence, tangent direction, constraint manifold.
- Check: positive definiteness, coordinate invariance, tractable projection, and discretization
  effects. A continuous invariant may disappear under the implemented step rule.

### Symmetry and equivariance

- Search for: group action, invariant/equivariant map, gauge symmetry, conservation via symmetry,
  quotient space.
- Target symptom: redundant representations or failure to generalize across known transformations.
- Check: the transformation is a true task symmetry and labels or dynamics transform as assumed.

## Statistical physics and thermodynamics

### Energy landscapes and free energy

- Search for: potential/free-energy minimization, Langevin dynamics, simulated annealing,
  metastability, barrier crossing.
- Target symptom: local trapping, noisy exploration, multiple stable states.
- Check: whether a valid scalar potential exists, what temperature/noise means in target units,
  and whether equilibrium assumptions are justified.

### Entropy, ensembles, and phase transitions

- Search for: maximum entropy, Gibbs distribution, order parameter, criticality, renormalization,
  universality class.
- Target symptom: diversity collapse, abrupt scaling behavior, many microscopic configurations
  with the same macrostate.
- Check: define microstates, constraints, and measure. “Entropy” without a distribution or counting
  rule is only a metaphor.

## Information and coding

### Compression and rate–distortion

- Search for: minimum description length, information bottleneck, rate–distortion, sufficient
  statistic, predictive coding.
- Target symptom: excessive representation, nuisance retention, resource-limited communication.
- Transported structure: source variable, representation/channel, distortion or task loss, rate.
- Check: what mutual information or code length is estimable and whether compression discards
  target-relevant information.

### Error correction and redundancy

- Search for: channel code, parity check, erasure code, fault tolerance, redundant sensing.
- Target symptom: corruption, component failure, unreliable communication.
- Check: independence and distribution of errors, code overhead, decoder availability, adversarial
  versus stochastic corruption.

## Optimization and online learning

### Duality, proximal structure, and splitting

- Search for: convex conjugate, Lagrangian dual, proximal operator, operator splitting, ADMM,
  primal-dual dynamics.
- Target symptom: coupled objectives, hard constraints, nonsmooth terms, decomposable subproblems.
- Check: convexity or monotonicity assumptions, duality gap, and whether each proximal step is
  computable.

### Regret, optimism, and adaptive preconditioning

- Search for: online convex optimization, no-regret learning, optimistic updates, adaptive
  regularization, second-order regret, preconditioning.
- Target symptom: nonstationarity, adversarial sequences, anisotropic gradients.
- Check: feedback model, comparator class, boundedness, and whether theoretical guarantees survive
  stochastic approximation.

## Biology and evolution

### Selection, variation, and ecological diversity

- Search for: mutation-selection dynamics, niche preservation, coevolution, frequency-dependent
  selection, bet hedging.
- Target symptom: premature convergence, environment change, portfolio collapse.
- Transported structure: population, variation operator, fitness/evaluator, inheritance, niches.
- Check: the target has repeated selection and a meaningful diversity descriptor. Biological terms
  alone add no mechanism.

### Homeostasis, adaptation, and modular regulation

- Search for: homeostatic regulation, allostasis, negative selection, regulatory network,
  modularity, robustness–evolvability.
- Target symptom: maintaining function under drift while avoiding self-damage.
- Check: sensed error, actuator, reference range, adaptation timescale, and failure under chronic
  perturbation.

## Networks, games, and markets

### Diffusion, topology, and collective dynamics

- Search for: consensus, epidemic threshold, percolation, graph diffusion, synchronization,
  message passing.
- Target symptom: propagation, bottlenecks, coordination, cascade risk.
- Check: edge meaning, direction, temporal topology, local versus global information, and threshold
  assumptions.

### Incentives and mechanism design

- Search for: incentive compatibility, signaling, auction, congestion game, potential game,
  principal-agent problem.
- Target symptom: components optimize local proxies that damage the system objective.
- Check: agent preferences, information, strategic adaptation, equilibrium concept, and whether
  agents are actually decision-makers rather than passive variables.

## Algorithms and computation

### Sketching, amortization, and lazy evaluation

- Search for: randomized sketch, streaming algorithm, amortized analysis, memoization, branch and
  bound, coarse-to-fine evaluation.
- Target symptom: memory, time, or evaluation cost dominates research iteration.
- Check: error bounds, adversarial inputs, update cost, and what information the approximation
  destroys.

### Evolutionary and population search

- Search for: island model, MAP-Elites, novelty search, quality diversity, crossover, mutation,
  multi-fidelity selection.
- Target symptom: large discrete design spaces with an automated evaluator and local optima.
- Check: evaluator fidelity, archive diversity, mutation locality, compute budget, and overfitting
  to the score.

Program-search systems such as [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) and
[AlphaEvolve](https://arxiv.org/abs/2506.13131) show why rich automated evaluators, executable
candidates, population memory, and diversity preservation are powerful when the task admits them.
They do not imply that every scientific question can be reduced to scalar program search.
