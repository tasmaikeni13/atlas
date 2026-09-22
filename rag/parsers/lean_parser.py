"""Lean 4 formal verification parser for mathematical proofs, theorems, and certificates."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from . import DocumentChunk

DECL_REGEX = re.compile(
    r"^(?:/--\s*(?P<doc>[\s\S]*?)-\/\s*)?"
    r"^(?P<kind>theorem|lemma|def|axiom)\s+(?P<name>[A-Za-z0-9_'.]+)",
    re.MULTILINE,
)

SECTION_REGEX = re.compile(r"^/-\!\s*(?P<section_doc>[\s\S]*?)-\/", re.MULTILINE)


def parse_lean_file(file_path: Path, repo_root: Path, category: str = "theory_proof") -> List[DocumentChunk]:
    """Parse Lean 4 source file into theorem and proof chunks with exact line numbers."""
    try:
        source_code = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    rel_path = str(file_path.relative_to(repo_root))
    lines = source_code.splitlines()
    total_lines = len(lines)
    chunks: List[DocumentChunk] = []

    # 1. Capture module-level comment block if present
    module_doc_match = re.search(r"^/-\s*\n(?P<header>[\s\S]*?)\n-\/", source_code)
    if module_doc_match:
        header_text = module_doc_match.group("header").strip()
        end_idx = source_code[: module_doc_match.end()].count("\n") + 1
        chunks.append(
            DocumentChunk(
                file_path=str(file_path),
                rel_path=rel_path,
                category=category,
                title=f"{rel_path} :: Formal Proof Overview",
                symbol=file_path.stem,
                start_line=1,
                end_line=end_idx,
                content=f"Formal Proof Module: {rel_path}\n{header_text}",
                metadata={"type": "lean_module_header"},
            )
        )

    # 2. Find all theorem / lemma / def declarations
    # Track positions of matches
    matches = list(re.finditer(r"(?:\n|^)(/--[\s\S]*?-\/\s*)?(theorem|lemma|def|axiom)\s+([A-Za-z0-9_'.]+)", source_code))

    for idx, match in enumerate(matches):
        decl_kind = match.group(2)
        decl_name = match.group(3)
        doc_comment = match.group(1) or ""

        start_char = match.start(1) if match.group(1) else match.start(2)
        start_line = source_code[:start_char].count("\n") + 1

        # End line is right before next declaration or EOF
        if idx + 1 < len(matches):
            next_start_char = matches[idx + 1].start(1) if matches[idx + 1].group(1) else matches[idx + 1].start(2)
            end_line = source_code[:next_start_char].count("\n")
        else:
            end_line = total_lines

        # Extract text block
        decl_content = "\n".join(lines[start_line - 1 : end_line]).strip()

        chunks.append(
            DocumentChunk(
                file_path=str(file_path),
                rel_path=rel_path,
                category=category,
                title=f"{rel_path} :: {decl_kind} {decl_name}",
                symbol=decl_name,
                start_line=start_line,
                end_line=end_line,
                content=f"-- Lean 4 Formal Verification: {decl_kind} {decl_name}\n{decl_content}",
                metadata={
                    "type": f"lean_{decl_kind}",
                    "name": decl_name,
                    "kind": decl_kind,
                    "has_doc": bool(doc_comment),
                },
            )
        )

    if not chunks:
        # Fallback to whole file if no declarations recognized
        chunks.append(
            DocumentChunk(
                file_path=str(file_path),
                rel_path=rel_path,
                category=category,
                title=f"{rel_path} :: Full Content",
                symbol=file_path.stem,
                start_line=1,
                end_line=total_lines,
                content=source_code[:3500],
                metadata={"type": "lean_file"},
            )
        )

    return chunks
