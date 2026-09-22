# Discovering Scientific Models from Data and Theory

Use this guide when the governing equation, causal structure, closure, or reduced-order model is
unknown. The objective is a model with predictive and explanatory value in a stated regime—not the
lowest training-error expression.

## 1. Define the model-discovery target

Specify:

- observables, latent variables, controls, and time/space coordinates;
- candidate model class: algebraic relation, ODE, PDE, stochastic process, causal graph, state-space
  model, or constitutive law;
- invariances, units, conservation relations, locality, causality, and boundary conditions;
- intended predictions and extrapolation regime;
- measurement and numerical uncertainty;
- minimum complexity and accuracy that would make the model useful.

Ask whether the available variables form a sufficient state. Non-Markovian residuals, hysteresis,
or unexplained dependence on history may signal missing state rather than a wrong coefficient.

## 2. Establish identifiability before searching expressions

Different models may generate indistinguishable observations under the existing experiment. Check:

- parameter and structural identifiability;
- collinearity among candidate terms;
- excitation of relevant modes;
- sampling rate and aliasing;
- observability of latent state;
- whether derivatives can be estimated at the noise level;
- whether boundary and initial conditions separate competing dynamics.

If models are observationally equivalent, design an intervention or new measurement before scaling
the search.

## 3. Construct a physically admissible hypothesis space

Use background knowledge to remove impossible expressions:

- dimensional consistency and nondimensional groups;
- symmetry and equivariance;
- conservation and balance laws;
- locality and causal direction;
- positivity, monotonicity, boundedness, or stability;
- known limiting cases and constitutive restrictions;
- sparse or modular structure only when scientifically plausible.

Keep a branch that relaxes each uncertain constraint. Incorrect “physics-informed” restrictions can
make the true model unreachable.

## 4. Choose the discovery engine by structure

### Symbolic regression

Use for compact algebraic relations or manageable expression grammars. Search a Pareto frontier of
fit versus complexity. Enforce units and known constraints. Refine coefficients after selecting
structure and test algebraic equivalence among expressions.

### Sparse system identification

Use when dynamics are sparse in a defensible function library. Estimate derivatives carefully or
use integral/weak formulations under noise. Sweep regularization and library composition; a sparse
answer is conditional on the basis offered.

### State-space and latent-variable models

Use when the measured variables are incomplete. Validate the latent state through prediction under
new inputs, interventions, or sensor combinations. Interpretability of a latent coordinate is a
hypothesis, not guaranteed by predictive fit.

### Mechanistic parameter estimation

Use when equations are known but parameters or closures are uncertain. Separate calibration from
validation. Inspect sloppiness, posterior correlations, multimodality, and parameter compensation.

### Causal structure discovery

Use only with explicit assumptions about interventions, time, hidden confounding, and measurement.
Return equivalence classes when data cannot orient edges. Predictive accuracy alone does not identify
causal direction.

## 5. Rank models on more than in-sample error

Use a model card:

```markdown
| Model | Fit/likelihood | Complexity | Physical constraints | Held-out prediction | Intervention prediction | Residual structure | Failure regime |
|---|---|---|---|---|---|---|---|
```

Inspect residual dependence on state, time, scale, and inputs. White residuals are not sufficient,
but structured residuals are strong evidence of missing dynamics or measurement effects.

## 6. Design experiments to separate surviving models

For each candidate condition, simulate or derive the predictive distribution under every live
model. Choose designs with large separation relative to measurement and parameter uncertainty,
subject to safety and cost. Consider:

- pulses, sweeps, and multisine or broadband excitation;
- initial conditions near different attractors or boundaries;
- scale changes and nondimensional regime transitions;
- interventions on suspected causal links;
- measurements of a latent mediator or conserved flux;
- a condition where one model predicts a null or sign reversal.

Expected information gain is useful only relative to the candidate model class. Include posterior
predictive or discrepancy checks that can reject every current model.

## 7. Validate discovery claims

A discovered law or mechanism needs:

- recovery on known synthetic or manufactured cases without leaking the answer;
- robustness to noise, sampling, and reasonable preprocessing;
- parameter and structure stability across resamples;
- prediction on untouched regimes or interventions;
- comparison with simpler empirical and established mechanistic models;
- explicit domain of validity;
- a derivation or physical interpretation when claiming mechanism;
- independent data or apparatus for a strong natural-law claim.

## Research basis

- SINDy discovers parsimonious dynamics by sparse selection from a candidate function library, with
  the basis and sparsity assumption made explicit: [Discovering governing equations from data by
  sparse identification of nonlinear dynamical systems](https://doi.org/10.1073/pnas.1517384113).
- Physics-inspired decomposition can exploit symmetry, separability, and compositionality in
  symbolic regression: [AI Feynman](https://pubmed.ncbi.nlm.nih.gov/32426452/).
- Noisy, incomplete experimental physics benefits from constraining the search with locality,
  causality, and spatial symmetries: [Physically constrained symbolic
  regression](https://www.nature.com/articles/s41467-021-23479-0).
- Similar data fits can be distinguished using background axioms and derivability:
  [AI-Descartes](https://www.nature.com/articles/s41467-023-37236-y).
- Bayesian experimental design formalizes experiment choice with utilities such as information gain:
  [Modern Bayesian Experimental Design](https://doi.org/10.1214/23-sts915).
