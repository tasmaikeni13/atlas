"""Parser for experiment outputs, benchmark metrics, and sharpness audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from . import DocumentChunk


def parse_run_json(file_path: Path, repo_root: Path, category: str = "runs") -> List[DocumentChunk]:
    """Parse JSON experiment metrics, benchmark results, and diagnostics into structured chunks."""
    try:
        raw_text = file_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except Exception:
        return []

    rel_path = str(file_path.relative_to(repo_root))
    lines = raw_text.splitlines()
    total_lines = len(lines)
    chunks: List[DocumentChunk] = []

    # Format human/agent-readable natural language summary of the metrics
    summary_parts = [f"Experiment Metric Artifact: {rel_path}"]
    
    if isinstance(data, dict):
        # Extract common keys
        for key, val in data.items():
            if isinstance(val, (int, float, str, bool)):
                summary_parts.append(f"- {key}: {val}")
            elif isinstance(val, list) and len(val) <= 10:
                summary_parts.append(f"- {key}: {val}")
            elif isinstance(val, dict):
                sub_keys = list(val.keys())[:8]
                summary_parts.append(f"- {key} (keys: {', '.join(sub_keys)})")

        # Specific handlers for known ATLAS runs
        if "inflation_factors" in data:
            summary_parts.append("\n**Sharpness Audit Highlights**:")
            summary_parts.append(f"Model Type: {data.get('model_type', 'unknown')}")
            summary_parts.append(f"True Sharpness: {data.get('true_sharpness')}")
            summary_parts.append(f"Batch Sizes: {data.get('batch_sizes')}")
            summary_parts.append(f"Inflation Factors: {data.get('inflation_factors')}")
            summary_parts.append(f"Discretization Step h: {data.get('discretization_h')}")

        if "spearman" in data or "atlas" in data:
            summary_parts.append("\n**Benchmark Convergence Highlights**:")
            if "atlas" in data and isinstance(data["atlas"], dict):
                atlas_res = data["atlas"]
                summary_parts.append(f"ATLAS Spearman Correlation: {atlas_res.get('spearman')}")
                summary_parts.append(f"ATLAS Relative L2 Error: {atlas_res.get('l2')}")
                summary_parts.append(f"ATLAS Sharpness Error: {atlas_res.get('sharpness_err')}")
                summary_parts.append(f"ATLAS Wall-clock Elapsed: {atlas_res.get('elapsed')}")

    content_str = "\n".join(summary_parts) + "\n\nRaw JSON snippet:\n" + raw_text[:2000]

    chunks.append(
        DocumentChunk(
            file_path=str(file_path),
            rel_path=rel_path,
            category=category,
            title=f"{rel_path} :: Metrics & Diagnostics",
            symbol=file_path.stem,
            start_line=1,
            end_line=total_lines,
            content=content_str,
            metadata={"type": "json_metrics", "stem": file_path.stem},
        )
    )

    return chunks
