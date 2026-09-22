"""CLI and programmatic query interface for ATLAS Research RAG."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.config import DB_PATH, REPO_ROOT
from rag.index import index_repository
from rag.store.db import RAGDatabase

CATEGORY_MAP = {
    "code": "code",
    "theory": "theory",
    "proof": "theory_proof",
    "paper": "theory_paper",
    "runs": "runs",
    "benchmark": "runs",
    "skill": "skill",
    "docs": "docs",
}


def search(
    query: str,
    category: Optional[str] = None,
    top_k: int = 5,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """Search the research index and return top matching chunks.
    
    If the index database does not exist, it will automatically build it.
    """
    if not db_path.exists():
        index_repository(db_path=db_path, repo_root=REPO_ROOT, verbose=False)

    db = RAGDatabase(db_path)
    cat_filter = CATEGORY_MAP.get(category.lower(), category) if category else None
    return db.search(query, category=cat_filter, top_k=top_k)


def format_markdown_results(query: str, results: List[Dict[str, Any]]) -> str:
    """Format results into agent-friendly GitHub-style Markdown with file links and line ranges."""
    if not results:
        return f"⚠️ No matching research artifacts found for query: `{query}`"

    lines = [
        f"### 🔍 ATLAS Research Retrieval: `{query}`",
        f"*Retrieved {len(results)} relevant artifact chunks:*\n",
    ]

    for idx, r in enumerate(results, 1):
        rel_path = r["rel_path"]
        abs_path = r["file_path"]
        start_l = r["start_line"]
        end_l = r["end_line"]
        cat = r["category"].upper()
        symbol = r.get("symbol") or "N/A"
        title = r["title"]
        score = r.get("rerank_score", r.get("rank", 0.0))

        # Markdown link with line number range
        file_link = f"[{rel_path}#L{start_l}-L{end_l}](file://{abs_path}#L{start_l}-L{end_l})"

        lines.append(f"#### {idx}. [{cat}] `{symbol}` in {file_link}")
        lines.append(f"- **Title**: {title}")
        lines.append(f"- **Relevance Score**: {score}")
        lines.append("\n```text")
        
        # Limit snippet lines to ~25 lines to keep context tight
        snippet_lines = r["content"].splitlines()
        if len(snippet_lines) > 28:
            snippet = "\n".join(snippet_lines[:28]) + f"\n... [+{len(snippet_lines) - 28} more lines in file]"
        else:
            snippet = "\n".join(snippet_lines)
            
        lines.append(snippet)
        lines.append("```\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search ATLAS research codebase, proofs, paper, benchmarks, and skills."
    )
    parser.add_argument("query", type=str, nargs="+", help="Research question, theorem name, symbol, or concept.")
    parser.add_argument(
        "--type",
        "-t",
        choices=["all", "code", "theory", "proof", "paper", "runs", "skill", "docs"],
        default="all",
        help="Filter results by category (default: all).",
    )
    parser.add_argument("--top", "-k", type=int, default=5, help="Number of results to return (default: 5).")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format.")

    args = parser.parse_args()
    full_query = " ".join(args.query)
    cat_filter = None if args.type == "all" else args.type

    results = search(full_query, category=cat_filter, top_k=args.top)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_markdown_results(full_query, results))


if __name__ == "__main__":
    main()
