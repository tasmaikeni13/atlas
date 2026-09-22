# Simulation, Measurement, and Uncertainty

This guide is especially relevant to physics, engineering, and other work in which a computational
model is compared with instruments or physical systems.

## 1. Model hierarchy

Maintain separate records for:

| Layer | Key question | Typical failure |
|---|---|---|
| Conceptual model | Are the relevant processes included? | missing physics, wrong system boundary |
| Mathematical model | Do equations and closures represent the concept? | invalid constitutive law or boundary condition |
| Numerical method | Does the discretization approximate the equations? | instability, truncation, solver error |
| Code | Is the method implemented correctly? | indexing, units, state, parallel nondeterminism |
| Inputs/parameters | Are values and distributions defensible? | calibration bias, nonidentifiability |
| Validation comparison | Does the model represent reality for intended use? | compensating errors, regime mismatch |

Calibration can make the output agree by compensating for model error. Use separate calibration and
validation observations where possible.

## 2. Dimensional and limiting checks

- verify unit consistency at every interface;
- non-dimensionalize to identify controlling groups and comparable regimes;
- derive zero, infinite, equilibrium, symmetry, and conservation limits;
- compare scale estimates before running a complex solver;
- check whether fitted parameters remain physically meaningful across scales.

A dimensionally correct equation may still be physically wrong, but a dimensionally inconsistent
one is wrong.

## 3. Code verification

Test the implementation against:

- unit and property tests for local operations;
- conservation or invariance checks;
- exact or analytic solutions;
- method of manufactured solutions when applicable;
- independent implementations or trusted benchmarks;
- deterministic small cases with hand-computable outcomes;
- restart, parallelization, and precision changes.

Code verification asks whether the algorithm was implemented as specified, not whether the
algorithm solves the intended physical model accurately.

## 4. Solution verification

Estimate numerical error using resolution, timestep, polynomial order, solver tolerance, particle
count, or domain-size studies as relevant. Seek the asymptotic convergence regime rather than
assuming the finest available run is accurate.

Record:

- convergence quantity and expected order;
- grids/resolutions and refinement ratios;
- iterative and discretization error estimates;
- sensitivity to boundary placement and initialization;
- stochastic simulation error and effective sample size;
- unresolved scales and subgrid/closure dependence.

If the quantity of interest does not converge, report that before comparing with experiment.

## 5. Measurement model

Define a measurand and forward measurement relation:

```text
raw indication = instrument(system quantity, environment, calibration, noise, drift)
reported value = reduction(raw indication, corrections, reference standards)
```

Inventory uncertainty sources:

- reference standard and calibration fit;
- resolution, quantization, threshold, saturation;
- repeatability and reproducibility across operators/instruments;
- sample preparation and positioning;
- environmental effects and drift;
- background subtraction and preprocessing;
- model-based correction and interpolation;
- spatial/temporal sampling and aliasing.

Separate random from systematic components only when the distinction is justified. Correlated
uncertainties must not be combined as independent.

## 6. Propagate uncertainty

Propagate the joint uncertainty distribution through the complete reduction and model comparison.
Use analytic propagation for well-behaved approximately linear relations; use Monte Carlo,
bootstrap, Bayesian, or interval methods when nonlinearities, bounds, or non-Gaussian sources
matter.

Report the interval, coverage or credibility interpretation, assumptions, and dominant contributors.
Use sensitivity analysis to decide which new calibration or measurement would reduce conclusion
uncertainty most.

NIST’s measurement guidance treats uncertainty as incomplete knowledge about the measurand and
provides methods for combining components: [Measurement Uncertainty](https://www.nist.gov/itl/sed/topic-areas/measurement-uncertainty).

## 7. Validation for intended use

Define the intended prediction and accuracy requirement before comparison. Use physical data that
exercise the relevant processes and were not used to tune free parameters. Compare at the level of
raw observables or through an explicit observation model so simulation and instrument outputs are
commensurate.

Possible outcomes:

- agreement within combined uncertainty for the tested regime;
- statistically clear discrepancy too small to matter for intended use;
- practically important model discrepancy;
- inconclusive because numerical or measurement uncertainty is too large;
- compensating agreement that fails another observable or regime.

Validation is graded and use-specific, not a permanent certificate for the model.

NIST distinguishes verification—accurately representing and solving the mathematical model—from
validation—representing the real world for intended use: [Summary of Industrial Verification,
Validation, and Uncertainty Quantification Procedures](https://nvlpubs.nist.gov/nistpubs/ir/2020/NIST.IR.8298.pdf).

## 8. Reproducibility versus replication

Retain data, code, parameters, environment, intermediate outputs, and nondeterministic seeds for
computational reproducibility. Independent data, apparatus, site, or team tests replicability and
may legitimately differ within system uncertainty.

The National Academies uses this distinction and notes that reproducing the same erroneous code is
not evidence of correctness: [Reproducibility and Replicability in Science](https://www.nationalacademies.org/read/25303/chapter/3).
