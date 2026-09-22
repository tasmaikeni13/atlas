"""Document parsers and chunkers for ATLAS Research RAG."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class DocumentChunk:
    """A semantic chunk of code, theory, proof, experiment data, or doc."""
    file_path: str
    rel_path: str
    category: str
    title: str
    symbol: str
    start_line: int
    end_line: int
    content: str
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "rel_path": self.rel_path,
            "category": self.category,
            "title": self.title,
            "symbol": self.symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "metadata": json.dumps(self.metadata),
        }
