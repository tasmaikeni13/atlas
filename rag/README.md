# 🧠 ATLAS Research RAG (Retrieval-Augmented Generation)

> **Agent Instruction**: This directory is the dedicated research retrieval and semantic index for the ATLAS project. 
> Whenever you need context regarding mathematical proofs, paper formulations, JAX code symbols, experiment runs, or research protocols, use the tools documented below before making assumptions or modifying files.

---

## ⚡ Agent Quickstart: CLI Commands

Run queries directly via bash in the workspace root:

```bash
# General search across the entire research repo
python3 -m rag.search "<your research question or symbol>"

# Category-specific searches
python3 -m rag.search "alloc_lower_bound" --type proof      # Lean 4 formal proofs
python3 -m rag.search "Hermite-Taylor partition" --type code # JAX / Python implementations
python3 -m rag.search "minimax budget rate" --type paper     # LaTeX paper theorems & equations
python3 -m rag.search "sharpness noise inflation" --type runs # Benchmark results & metrics JSON
python3 -m rag.search "falsifiable learning loop" --type skill # Research methodology & protocols

# Machine-readable JSON output (for automated tool chaining)
python3 -m rag.search "DKW certificate" --json

# Rebuild the index after modifying code, proofs, or paper
python3 -m rag.index
```

---

## 📚 What This RAG System Covers

| Category Flag | Target Path | Description |
|---|---|---|
| `--type proof` | [`proofs/AtlasCert/`](file:///home/tasma/stam/proofs/AtlasCert) | Lean 4 formal certificates (`alloc_lower_bound`, `debias_unbiased`, etc.) |
| `--type paper` | [`paper/`](file:///home/tasma/stam/paper) | Academic LaTeX paper ([atlas.tex](file:///home/tasma/stam/paper/atlas.tex)) with theorems, definitions, and equations |
| `--type code` | [`atlas/`](file:///home/tasma/stam/atlas) | JAX/Flax TPU implementations, Taylor jet probes, basis projection, reconstruction |
| `--type runs` | [`runs/`](file:///home/tasma/stam/runs) | Empirical benchmark metrics, sharpness audits, 125M trajectories, diagnostic JSONs |
| `--type skill` | [`skills/`](file:///home/tasma/stam/skills) | Operating rules and research protocols (ML research, theory attack, study design) |
| `--type all` | All of the above | Default cross-modal search spanning theory, proofs, code, and empirical logs |

---

## 🎯 Example Agent Queries & Use Cases

### 1. Connecting Math/Theory to Lean 4 Formal Proofs
* **Query:** `python3 -m rag.search "minimax budget allocation lower bound" --type proof`
* **Result:** Pulls theorem [`alloc_lower_bound`](file:///home/tasma/stam/proofs/AtlasCert/AtlasCert/Certificates.lean#L34-L56) in Lean 4 with full tactic proof showing the weighted AM-GM inequality bound.

### 2. Checking Empirical Benchmark Numbers & Baselines
* **Query:** `python3 -m rag.search "sharpness inflation factor" --type runs`
* **Result:** Pulls exact metrics from [`runs/sharpness/sharpness_audit_transformer.json`](file:///home/tasma/stam/runs/sharpness/sharpness_audit_transformer.json) showing $11\times$ to $21.5\times$ curvature noise inflation on finite-difference baselines.

### 3. Locating Core Algorithms in JAX
* **Query:** `python3 -m rag.search "HermiteTaylorReconstruction partition of unity" --type code`
* **Result:** Pulls the class definition and evaluation batch methods in [`atlas/reconstruct.py`](file:///home/tasma/stam/atlas/reconstruct.py#L35-L139).

---

## 🐍 Programmatic Python API

Any agent or script can import and query the index directly in Python:

```python
from rag import search

# Query top 5 results
results = search("DKW certification anchor bound", category="theory", top_k=5)

for r in results:
    print(f"[{r['category']}] {r['title']}")
    print(f"File: {r['rel_path']} (Lines {r['start_line']}-{r['end_line']})")
    print(r["content"][:200])
```

---

## 🔄 Index Maintenance

If you edit or add any `.py`, `.lean`, `.tex`, `.json`, or `.md` files in the repository:
```bash
python3 -m rag.index
```
The indexing pipeline runs incrementally in ~0.15 seconds, parsing ASTs, formal proof declarations, and LaTeX sections.
