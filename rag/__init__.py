"""ATLAS Research RAG: Domain-aware retrieval system for Code, Lean 4 Proofs, Paper, and Runs."""

from typing import Any, Dict, List, Optional


def search(*args, **kwargs) -> List[Dict[str, Any]]:
    """Search the research index and return top matching chunks."""
    from rag.search import search as _search
    return _search(*args, **kwargs)


def index_repository(*args, **kwargs) -> int:
    """Index or re-index the entire research repository into SQLite FTS5 store."""
    from rag.index import index_repository as _index_repository
    return _index_repository(*args, **kwargs)


__all__ = ["search", "index_repository"]
