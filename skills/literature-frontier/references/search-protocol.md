# Search Protocol for Frontier and Novelty Work

Use this protocol when the answer depends on coverage rather than a few known sources. The goal is
high recall first, then precise comparison—not a single clever query.

## 1. Expand the research contract into a query lattice

Create query terms along independent axes:

| Axis | Generate terms for |
|---|---|
| Target | phenomenon, task, population, dataset, theorem, system |
| Operation | predict, optimize, estimate, control, compress, prove, generate |
| Failure | instability, bias, variance, brittleness, cost, impossibility, leakage |
| Mechanism | update rule, representation, objective, feedback, invariant, geometry |
| Evidence | benchmark, ablation, counterexample, replication, negative result, survey |
| Terminology | acronyms, historical names, adjacent-field names, mathematical forms |

Run combinations that cover different axes. Preserve exact phrases only for distinctive names or
equations; otherwise use several paraphrases. Add date filters only after collecting the historical
lineage, because recent work may assume older terminology.

## 2. Use four complementary search directions

### Target-forward

Start from the problem and retrieve methods, comparisons, failures, and open questions.

### Mechanism-backward

Start from a candidate mechanism or equation and search for target applications, synonymous
implementations, and negative results.

### Citation-graph traversal

For every closest paper:

- read the works it treats as immediate predecessors;
- inspect later papers that cite it for replications, corrections, and extensions;
- inspect authors’ adjacent work and released code when implementation details matter;
- follow benchmark or dataset papers separately from method papers.

### Disconnected-literature bridging

Search chains of the form `target A -> intermediate relation B -> source C`. The bridge term must
be a concrete variable, process, or mathematical structure. A shared buzzword is not a bridge.
Confirm both A–B and B–C links in primary sources before considering the implied A–C hypothesis.

## 3. Separate discovery sources from evidence sources

Discovery aids include search engines, surveys, review articles, paper recommendation systems,
blogs, and snippets. They can point to evidence but should not silently become the evidence.

Prefer, in descending order when appropriate:

1. original paper plus official artifact or code;
2. independent replication, benchmark, or reanalysis;
3. official documentation or dataset specification;
4. peer-reviewed synthesis;
5. preprint with accessible methods and artifacts;
6. commentary only for context or leads.

Venue prestige and citation count are not substitutes for methodological fit. Record retractions,
errata, and version dates.

## 4. Extract facets before comparing novelty

Represent both the proposed idea and each candidate prior work by the same facets:

- problem and regime;
- inputs and outputs;
- representation or state;
- transformation, objective, or update rule;
- training or solution procedure;
- resource assumptions;
- predicted behavior or guarantee;
- evaluation protocol;
- composition of components.

Retrieve broadly using keywords, snippets, equations, and synonyms. Then rerank manually by facet
overlap. Semantic similarity alone can miss equivalent methods with different language and can rank
surface-similar but structurally different work too highly.

Use one of these overlap labels:

- **same** — the load-bearing facets and operating regime match;
- **contained** — one method is a special case of the other;
- **component precedent** — a key part is known but the composition may differ;
- **analogous** — relation structure is similar but target objects or assumptions differ;
- **adjacent** — same problem, different mechanism, or vice versa;
- **unrelated after inspection**.

## 5. Search for disconfirmation deliberately

For every promising gap, issue queries designed to destroy it:

- exact equations or pseudocode fragments;
- older terminology and names of precursor fields;
- “negative result,” “fails,” “limitations,” “reappraisal,” “replication,” or
  “counterexample” with the key mechanism;
- patents, theses, workshop papers, technical reports, and code repositories when relevant;
- the proposed component pair in both orders;
- the behavior the method predicts, without the method name.

Read the closest disconfirming source before ranking the opportunity.

## 6. Know when the search is saturated

Coverage is provisionally adequate when:

- several independent query families converge on the same closest work;
- backward and forward citation traversal stop adding new method families;
- the comparison matrix has primary evidence for every load-bearing cell;
- new sources mostly duplicate already extracted mechanisms or results;
- the remaining uncertainty is named and does not decide the top opportunity.

Do not use an arbitrary paper count. A narrow theorem may saturate with few sources; a fragmented
applied field may require many.

## 7. Source basis for this protocol

- Swanson’s literature-based discovery work demonstrated that complementary A–B and B–C findings
  can remain disconnected across literatures: [Fish oil, Raynaud's syndrome, and undiscovered
  public knowledge](https://pubmed.ncbi.nlm.nih.gov/3797213/).
- Gentner’s structure-mapping account distinguishes relational structure from shared attributes:
  [Structure-Mapping: A Theoretical Framework for Analogy](https://onlinelibrary.wiley.com/doi/epdf/10.1207/s15516709cog0702_3).
- Facet-aware retrieve-then-rerank improved literature-grounded idea novelty judgments in
  [Literature-Grounded Novelty Assessment of Scientific Ideas](https://aclanthology.org/2025.sdp-1.9/).
- ResearchBench decomposes discovery into inspiration retrieval, hypothesis composition, and
  ranking: [ResearchBench](https://arxiv.org/abs/2503.21248).
- Scideator reports benefits from paper-facet recombination and explicit novelty search:
  [Scideator](https://arxiv.org/abs/2409.14634).

These sources motivate the operations above; they do not guarantee that any individual gap or
analogy is novel.
