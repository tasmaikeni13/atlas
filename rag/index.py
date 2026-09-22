"""Repository indexing pipeline that parses all research code, proofs, paper, and runs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

from rag.config import DB_PATH, IGNORE_PATTERNS, INDEX_TARGETS, REPO_ROOT, SUPPORTED_EXTENSIONS
from rag.parsers import DocumentChunk
from rag.parsers.latex_parser import parse_latex_file
from rag.parsers.lean_parser import parse_lean_file
from rag.parsers.md_parser import parse_markdown_file
from rag.parsers.python_parser import parse_python_file
from rag.parsers.run_parser import parse_run_json
from rag.store.db import RAGDatabase


def is_ignored(path: Path) -> bool:
    """Check if file or path matches ignore list."""
    path_str = str(path)
    for pattern in IGNORE_PATTERNS:
        if pattern in path_str:
            return True
    return False


def collect_target_files(repo_root: Path) -> List[tuple[Path, str]]:
    """Collect all eligible files in the repository mapped to their category."""
    collected: List[tuple[Path, str]] = []

    for target_name, category in INDEX_TARGETS.items():
        target_path = repo_root / target_name
        if not target_path.exists():
            continue

        if target_path.is_file():
            if target_path.suffix in SUPPORTED_EXTENSIONS and not is_ignored(target_path):
                collected.append((target_path, category))
        elif target_path.is_dir():
            for p in target_path.rglob("*"):
                if p.is_file() and p.suffix in SUPPORTED_EXTENSIONS and not is_ignored(p):
                    collected.append((p, category))

    return collected


def parse_file(file_path: Path, repo_root: Path, category: str) -> List[DocumentChunk]:
    """Dispatch file parsing to corresponding domain parser."""
    suffix = file_path.suffix.lower()

    if suffix == ".py":
        return parse_python_file(file_path, repo_root, category=category)
    elif suffix == ".lean":
        return parse_lean_file(file_path, repo_root, category=category)
    elif suffix in (".tex", ".bib"):
        return parse_latex_file(file_path, repo_root, category=category)
    elif suffix == ".json":
        return parse_run_json(file_path, repo_root, category=category)
    elif suffix == ".md":
        return parse_markdown_file(file_path, repo_root, category=category)
    else:
        return []


def index_repository(db_path: Path = DB_PATH, repo_root: Path = REPO_ROOT, verbose: bool = True) -> int:
    """Index or re-index the entire research repository into SQLite FTS5 store."""
    t0 = time.time()
    db = RAGDatabase(db_path)
    db.clear()

    files = collect_target_files(repo_root)
    all_chunks: List[DocumentChunk] = []

    for file_path, category in files:
        chunks = parse_file(file_path, repo_root, category)
        all_chunks.extend(chunks)

    inserted_count = db.insert_chunks(all_chunks)
    elapsed = time.time() - t0

    if verbose:
        stats = db.get_stats()
        print(f"✅ Indexed {inserted_count} chunks from {len(files)} files in {elapsed:.3f}s")
        print("📊 Breakdown by category:")
        for cat, cnt in sorted(stats.items()):
            print(f"   • {cat:<18} : {cnt:>4} chunks")

    return inserted_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the ATLAS Research repository into SQLite FTS5 RAG.")
    parser.add_argument("--force", action="store_true", help="Force fresh rebuild of the database.")
    args = parser.parse_args()

    print(f"🚀 Indexing ATLAS Research Repository at: {REPO_ROOT}")
    total = index_repository()
    print(f"✨ Ready! Database located at: {DB_PATH}")


if __name__ == "__main__":
    main()
