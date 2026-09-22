"""Configuration and paths for ATLAS Research RAG."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
RAG_DIR: Path = REPO_ROOT / "rag"
DB_PATH: Path = RAG_DIR / "index.db"

# Subdirectories and targets to index with their respective category tag
INDEX_TARGETS: Dict[str, str] = {
    "atlas": "code",
    "proofs": "theory_proof",
    "paper": "theory_paper",
    "experiments": "code_experiment",
    "runs": "runs",
    "skills": "skill",
    "README.md": "docs",
}

# Supported file extensions
SUPPORTED_EXTENSIONS: Set[str] = {
    ".py",
    ".lean",
    ".tex",
    ".bib",
    ".json",
    ".md",
}

# Paths and patterns to explicitly ignore
IGNORE_PATTERNS: Set[str] = {
    "__pycache__",
    ".git",
    ".lake",
    "lake-packages",
    ".egg-info",
    ".venv",
    "node_modules",
    "rag/index.db",
    "rag/index.db-journal",
    ".system_generated",
}

# Maximum characters per chunk to prevent context overflow while keeping full semantic blocks
MAX_CHUNK_CHARS: int = 3500
