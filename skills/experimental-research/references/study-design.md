# Study Design for Empirical Research

Choose a design that can distinguish the live models under the actual constraints. Statistical
sophistication cannot repair a design that confounds the treatment with time, batch, subject, or
instrument.

## 1. Identify the experimental unit

The experimental unit is the smallest independently assigned or sampled unit. Repeated readings,
pixels, cells from the same culture, time points from one trajectory, and subsamples from one
physical specimen usually do not create independent replication.

Record the hierarchy:

```text
site -> instrument/operator -> batch -> experimental unit -> repeated measurement
```

Model or aggregate dependencies at the correct level. Avoid pseudoreplication.

## 2. Select the design family

### Controlled intervention

Use when the causal factor can be assigned. Include a negative control, a positive control when
available, and a baseline/sham condition. Randomize assignment and run order. Blind measurement or
analysis where feasible.

### Factorial design

Use when interactions matter or several factors can be varied efficiently. Start with scientifically
plausible ranges. Include center points or replication when curvature and noise must be separated.
Do not interpret main effects without checking strong interactions.

### Blocking or matched design

Use when known nuisance variation—batch, day, site, specimen, baseline state—is large. Block before
randomization. A block should be related to outcome variation but not created by treatment.

### Longitudinal or repeated-measures design

Use for trajectories and within-unit changes. Model autocorrelation, carryover, maturation, and
dropout. Counterbalance order when previous interventions affect later responses.

### Observational design

Use when assignment is impossible or unethical. Draw a causal diagram before adjustment. Measure
confounders that jointly affect exposure and outcome. Do not control colliders or post-treatment
variables without a justified estimand. Use negative controls, sensitivity analysis, natural
experiments, or instrumental variables only when their assumptions can be defended.

### Sequential or adaptive design

Use when measurements are costly and interim data can select the next condition. Predefine the
adaptation and stopping rule for confirmatory inference. Track all tested conditions to avoid
selective reporting.

When live mechanistic models exist, score candidate experiments by how differently they predict the
outcome, expected reduction in decision-relevant uncertainty, cost, and safety. Add a model-checking
condition that can expose shared misspecification rather than only discriminating within the current
set.

## 3. Design from predictions, not from available columns

Create a prediction table:

```markdown
| Condition | Model A predicts | Model B predicts | Artifact/null predicts | Measurement uncertainty | Decision value |
|---|---|---|---|---|---|
```

Prioritize conditions with large prediction separation relative to total uncertainty. Add boundary
conditions where models predict disappearance, reversal, saturation, or a different scaling law.

## 4. Define controls

- **Negative control:** should not respond through the proposed mechanism; detects contamination,
  drift, placebo, or analysis artifacts.
- **Positive control:** known response confirms that apparatus and protocol can detect an effect.
- **Sham/procedural control:** matches handling without the active intervention.
- **Calibration/reference:** known quantity tests the measurement chain.
- **Mechanism control:** removes or blocks a necessary link while holding other changes fixed.
- **Alternative-explanation control:** recreates the confound without the proposed cause.

Controls need their own expected ranges and failure actions.

## 5. Size the study for the scientific decision

Specify:

- minimum effect that would change the model or decision;
- expected within- and between-unit variation;
- measurement uncertainty and failure/dropout rate;
- number of factors, comparisons, and planned stopping looks;
- desired interval width or probability of detecting/excluding the meaningful effect.

Use simulation-based design analysis for hierarchical, nonlinear, censored, or adaptive models.
Report sensitivity to uncertain variance and effect assumptions. “Powered at 80%” without these
inputs is not informative.

## 6. Predefine the confirmatory analysis

Freeze:

- primary response and contrast;
- inclusion, exclusion, and missing-data rules;
- transformation and normalization;
- statistical model and uncertainty interval;
- handling of multiple outcomes or comparisons;
- stopping and model-selection rule;
- diagnostic failures that invalidate the run.

Exploratory analyses remain valuable when labeled and confirmed later.

## 7. Pilot correctly

Use a pilot to test feasibility, calibration, range, protocol compliance, variance scale, and data
pipeline. A pilot is not a miniature definitive study. Changes motivated by pilot outcomes create a
new protocol; confirmation needs untouched evidence.

## 8. Design quality audit

- Can each treatment or exposure be separated from batch, order, site, and operator?
- Is the experimental unit truly independent?
- Do controls diagnose the main artifact pathways?
- Can the instrument resolve the minimum meaningful difference?
- Do competing models make observably different predictions in tested conditions?
- Are the planned analyses compatible with the sampling and assignment process?
- Is the stopping rule independent of the desired conclusion?
- Are ethical and safety constraints met?

NIST describes design of experiments as a way to obtain valid, defensible conclusions while using
experimental resources efficiently: [What is experimental design?](https://www.itl.nist.gov/div898/handbook/pri/section1/pri11.htm)
and [What is design of experiments?](https://itl.nist.gov/div898/handbook/pmd/section3/pmd31.htm).

Closed-loop scientific systems provide direct evidence that selection policy matters. The early
Robot Scientist’s model-based selection reduced experiment cost relative to cheapest and random
selection: [Functional genomic hypothesis generation and experimentation by a robot
scientist](https://www.nature.com/articles/nature02236). Recent systems likewise couple literature,
hypotheses, experiments, and updated analysis rather than treating ideation as completion:
[Robin](https://www.nature.com/articles/s41586-026-10652-y) and
[A-Lab](https://www.nature.com/articles/s41586-023-06734-w).
