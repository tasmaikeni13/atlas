---
name: literature-frontier
description: Map a research field from primary literature, establish the closest prior work, test whether an idea is already known, and turn genuine evidence gaps into citation-backed research opportunities. Use for state-of-the-art reviews, research-gap searches, novelty checks, benchmark or method landscapes, and open-question discovery; do not use merely to summarize a supplied paper.
---

# Literature Frontier

Produce a defensible map of what is known, what is uncertain, and what remains worth testing. The
result is not a reading list. It is a set of source-linked claims, comparisons, contradictions, and
research opportunities that another researcher can audit.

## Non-negotiable standards

- Search current literature. Prefer original papers, official artifacts, datasets, and code over
  surveys or commentary. Use reviews to learn vocabulary and locate primary work, not as the sole
  evidence for technical claims.
- Read the papers closest to the proposed contribution. Titles, abstracts, snippets, and citation
  counts are discovery aids, not sufficient evidence for novelty or method details.
- Separate four labels: **reported by source**, **replicated independently**, **inferred from
  sources**, and **open speculation**.
- Treat “no paper found” as weak evidence. Novelty requires a deliberate search for synonyms,
  equivalent formulations, component precedents, and work in neighboring fields.
- Never manufacture an open problem from an empty search result. A useful gap needs a reason the
  gap matters, evidence it is not already closed, and a feasible observation that would reduce the
  uncertainty.

## Working loop

### 1. Write the research contract

Before searching, fix:

- target phenomenon or capability;
- object of study and operating regime;
- outcome or claim that would count as progress;
- comparison class, including the strongest plausible prior baseline;
- constraints such as compute, data, latency, theory, safety, or deployment;
- time boundary for the search and any user-specified sources.

If the request is vague, formulate the narrowest useful contract consistent with it and mark the
assumptions. Do not silently replace the requested problem with an easier literature question.

### 2. Build and execute a query lattice

Read [references/search-protocol.md](references/search-protocol.md) before a serious frontier or
novelty search. Expand the query along the problem, mechanism, failure, measurement, and terminology
axes. Search both forward from the target problem and backward from candidate mechanisms. Traverse
references and citing work around the closest papers.

Keep a search ledger. Record productive queries, databases or sites, coverage dates, and unresolved
terminology. This prevents repeated shallow searches from masquerading as breadth.

### 3. Extract evidence, not summaries

For each load-bearing paper, record:

- exact question and claimed contribution;
- method or mechanism in operational terms;
- assumptions, data, evaluation regime, and comparison budget;
- result magnitude and uncertainty when reported;
- ablations, negative results, and known failure boundary;
- artifact availability and whether the result has independent support;
- the precise claim for which this paper is evidence.

Do not infer a method’s behavior outside its evaluated regime without labeling that move as an
inference.

### 4. Construct the frontier map

Organize the evidence into:

1. a closest-work matrix with rows for methods and columns for the dimensions that actually
   determine equivalence;
2. a chronology or citation lineage for the load-bearing ideas;
3. a benchmark and evaluation map showing where comparisons are and are not commensurate;
4. a contradiction and anomaly table;
5. a saturation map: well-covered cells, failed cells, and genuinely unresolved cells.

Choose axes from the evidence. Do not force generic axes onto a field. Apparent white space may be
an uninteresting regime, an equivalent formulation, an infeasible experiment, or a repeatedly
failed direction.

### 5. Turn gaps into testable opportunity tickets

A candidate gap is ready only when it specifies:

- the unresolved claim or missing capability;
- the closest work and exact technical difference;
- evidence that the difference is not merely vocabulary;
- why resolving it would matter;
- the smallest decisive experiment, derivation, or counterexample;
- the strongest likely reason it will fail;
- a novelty falsifier: what prior result would collapse the claim.

Rank tickets by expected information gain and scientific value, not by how grand they sound. A
negative experiment that resolves a live uncertainty can be more valuable than a speculative new
architecture.

### 6. Run the adversarial novelty pass

Concretize the proposed idea before checking novelty. Search its components, composition, objective,
update rule, mathematical form, and predicted behavior separately. Retrieve broadly, then compare
the closest results facet by facet. Search for patents or non-paper artifacts when relevant.

Use only calibrated outcomes:

- **overlap found** — cite it and state whether anything remains;
- **no close overlap found within searched scope** — state databases, terms, and date;
- **novelty unresolved** — name the missing coverage or inaccessible source;
- **apparently distinct** — identify the nearest work and exact difference, without claiming a
  breakthrough.

## Required deliverable

Return or maintain the smallest set of artifacts that makes the analysis auditable:

- research contract;
- search ledger;
- evidence ledger;
- closest-work/frontier matrix;
- 1–5 ranked opportunity tickets;
- limitations and the next search or experiment that would most change the ranking.

Use the templates in [references/frontier-artifacts.md](references/frontier-artifacts.md) when the
work spans multiple sources or turns. Persist them in the user’s project when asked to conduct an
ongoing research program.

## Stop conditions

Stop when the relevant frontier is saturated enough that new searches mostly recover already mapped
work, the closest competitors have been read, and each proposed gap has a decisive next test. If
access, terminology, or evidence prevents that standard, report the exact missing coverage rather
than filling it with conjecture.
