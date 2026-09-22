"""Heuristic and lexical reranker to boost exact symbol hits, proximity, and title matches."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List


def rerank_results(query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Rerank FTS5 candidate results using exact symbol matching, title relevance, and n-gram overlap."""
    if not candidates:
        return []

    tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 1]
    query_phrase = " ".join(tokens)

    scored: List[tuple[float, Dict[str, Any]]] = []

    for item in candidates:
        base_bm25 = -float(item.get("rank", 0.0))  # SQLite bm25 returns negative values where lower is better
        score = base_bm25

        title_lower = (item.get("title") or "").lower()
        symbol_lower = (item.get("symbol") or "").lower()
        content_lower = (item.get("content") or "").lower()
        rel_path_lower = (item.get("rel_path") or "").lower()

        # 1. Boost exact symbol matches
        for t in tokens:
            if t == symbol_lower or f".{t}" in symbol_lower or f"_{t}" in symbol_lower:
                score += 8.0
            if t in title_lower:
                score += 3.0
            if t in rel_path_lower:
                score += 2.0

        # 2. Boost exact full query phrase match
        if query_phrase and query_phrase in content_lower:
            score += 10.0
        if query_phrase and query_phrase in title_lower:
            score += 15.0

        # 3. Lean 4 theorem / lemma exact match boost
        if item.get("category") == "theory_proof" and any(t in symbol_lower for t in tokens):
            score += 5.0

        # 4. Code definition boost if asking for functions/classes
        if item.get("category") == "code" and any(t in symbol_lower for t in tokens):
            score += 4.0

        item["rerank_score"] = round(score, 3)
        scored.append((score, item))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]
