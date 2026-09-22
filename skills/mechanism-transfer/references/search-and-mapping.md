# Search and Mapping Protocol

This protocol turns a target failure into source mechanisms, then filters mappings before expensive
implementation.

## A. Generate search descriptions from the target

Write the target in five forms:

1. **behavioral** — what observable behavior is missing or excessive;
2. **dynamical** — how state evolves, including delays, feedback, noise, and timescales;
3. **constraint** — what is conserved, bounded, unavailable, or expensive;
4. **informational** — what is observed, forgotten, compressed, estimated, or routed;
5. **geometric/relational** — what distances, symmetries, orderings, or interactions matter.

Extract domain-neutral verbs and relations. Search phrases such as “stabilizes delayed noisy
feedback,” “online adaptation under partial observation,” or “constrained flow preserves
invariant,” combined with mechanism families and mathematical terms. Do not begin with “physics
inspired neural network”; it over-retrieves branding and under-retrieves the actual structure.

## B. Search near and far without becoming random

Use a query matrix:

| Structural job | Near-domain term | Mathematical term | Distant-domain realization | Failure/limit term |
|---|---|---|---|---|

Near sources reduce interface risk. Distant sources can reveal alternate implementations. A source
is worth reading only if it shares at least one load-bearing relation with the mechanism request.
Semantic distance by itself is not valuable.

Use citation chains and A–B–C bridges:

- A: target failure or variable;
- B: shared mediator, invariant, or relational structure;
- C: source mechanism controlling B.

Confirm A–B and B–C independently. Then treat A–C as a hypothesis, never as a deduction unless the
necessary causal assumptions are established.

## C. Compare multiple source analogues

When feasible, find two source systems that produce the desired behavior through the same proposed
relation but differ in surface features. Align them and extract what remains common. This
contrastive step exposes incidental source details and makes spontaneous far transfer more likely.

Do not force a single schema if the sources implement genuinely different mechanisms. Preserve
them as separate transfer branches.

## D. Mechanism card

```markdown
# [Mechanism name]
Primary source:
Behavior explained:
State variables and types:
Governing relation / causal chain:
Intervention or removal evidence:
Invariant or conserved/bounded quantity:
Required observations and controls:
Timescale and scale:
Boundary and known failures:
Domain-free form:
Confidence and unresolved causality:
```

The domain-free form should be an equation, typed transformation, state transition, causal graph,
or precise algorithm whenever possible.

## E. Relational mapping rules

1. Map relations and roles before object attributes.
2. Preserve causal direction. A source effect cannot become a target cause without a new argument.
3. Preserve higher-order structure: what modulates, constrains, or measures each relation.
4. Enforce one-to-one role consistency unless the adapter explicitly merges or splits roles.
5. Track unmapped source elements and target elements; both may invalidate the transfer.
6. Check units, types, dimensions, timescales, topology, and stochastic assumptions.
7. Translate into target-native terms only after the domain-neutral bridge is stable.

## F. Compatibility matrix

```markdown
| Requirement | Source provides | Target analogue | Evidence it holds | Mismatch | Adapter | New assumption |
|---|---|---|---|---|---|---|
```

Fatal mismatches include an unobservable required state, an unavailable intervention, reversed
causality, contradictory conservation/dissipation assumptions, or a timescale separation that the
target lacks. More parameters or a learned adapter do not automatically repair a structural
mismatch.

## G. Derive predictions before implementation

From the bridge model, derive:

- a positive signature in the intended regime;
- a null, disappearance, or reversal signature outside it;
- a scaling law or monotonic relationship if the mechanism supports one;
- a measurable internal quantity tied to the proposed mechanism;
- a simple confound that could mimic the headline metric.

Predictions determine the experiment. If all plausible results are compatible with the story, the
story is not yet scientific.

## H. Minimum transfer report

```markdown
Target socket:
Mechanism request:
Source portfolio and citations:
Chosen mechanism and why:
Relational map:
Bridge model:
Target reconstruction:
Compatibility verdict:
Risky prediction:
Cheapest kill test:
Novelty-search scope:
Status: rejected | conceptual | implemented | survived screen | confirmed
```

## Research basis

- Relational and higher-order structure, not shared object attributes, is central to a sound
  analogy in [Structure-Mapping: A Theoretical Framework for Analogy](https://onlinelibrary.wiley.com/doi/epdf/10.1207/s15516709cog0702_3).
- People often fail to retrieve a distant analogy without a cue, even when it is useful:
  [Analogical problem solving](https://www.sciencedirect.com/science/article/pii/0010028580900134).
- Comparing analogues and explicitly stating the shared schema improves transfer:
  [Schema Induction and Analogical Transfer](https://reasoninglab.psych.ucla.edu/wp-content/uploads/sites/273/2021/04/Gick_Holyoak1983_SchemaInduction.pdf).
- Swanson’s work motivates searching logically connected but bibliographically separated
  literatures: [Fish oil, Raynaud's syndrome, and undiscovered public knowledge](https://pubmed.ncbi.nlm.nih.gov/3797213/).
- Large-scale bibliometric evidence associates high impact with a conventional knowledge base plus
  a limited atypical combination—not novelty everywhere: [Atypical combinations and scientific
  impact](https://pubmed.ncbi.nlm.nih.gov/24159044/).
