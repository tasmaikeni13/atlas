"""Parser for Markdown documentation, skill files, and operating protocols."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from . import DocumentChunk

HEADER_REGEX = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
FRONTMATTER_REGEX = re.compile(r"^---\s*\n(?P<fm>[\s\S]*?)\n---\s*\n")


def parse_markdown_file(file_path: Path, repo_root: Path, category: str = "skill") -> List[DocumentChunk]:
    """Parse a Markdown file by section headers preserving YAML frontmatter and line numbers."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    rel_path = str(file_path.relative_to(repo_root))
    lines = content.splitlines()
    total_lines = len(lines)
    chunks: List[DocumentChunk] = []

    # Check for frontmatter
    fm_match = FRONTMATTER_REGEX.match(content)
    frontmatter_text = ""
    offset_chars = 0
    if fm_match:
        frontmatter_text = fm_match.group("fm").strip()
        offset_chars = fm_match.end()
        fm_end_line = content[:offset_chars].count("\n") + 1
        chunks.append(
            DocumentChunk(
                file_path=str(file_path),
                rel_path=rel_path,
                category=category,
                title=f"{rel_path} :: Skill Frontmatter & Metadata",
                symbol=file_path.stem,
                start_line=1,
                end_line=fm_end_line,
                content=f"Skill Frontmatter:\n{frontmatter_text}",
                metadata={"type": "frontmatter"},
            )
        )

    # Find headers
    matches = list(HEADER_REGEX.finditer(content))
    if not matches:
        chunks.append(
            DocumentChunk(
                file_path=str(file_path),
                rel_path=rel_path,
                category=category,
                title=f"{rel_path} :: Full Content",
                symbol=file_path.stem,
                start_line=1,
                end_line=total_lines,
                content=content[:3500],
                metadata={"type": "markdown_doc"},
            )
        )
        return chunks

    for idx, match in enumerate(matches):
        header_level = len(match.group(1))
        header_title = match.group(2).strip()

        start_char = match.start()
        start_line = content[:start_char].count("\n") + 1

        if idx + 1 < len(matches):
            next_start_char = matches[idx + 1].start()
            end_line = content[:next_start_char].count("\n")
        else:
            end_line = total_lines

        chunk_lines = lines[start_line - 1 : end_line]
        chunk_text = "\n".join(chunk_lines).strip()

        chunks.append(
            DocumentChunk(
                file_path=str(file_path),
                rel_path=rel_path,
                category=category,
                title=f"{rel_path} :: {header_title}",
                symbol=header_title,
                start_line=start_line,
                end_line=end_line,
                content=f"Documentation Section: {header_title}\n\n{chunk_text}",
                metadata={"type": "md_section", "level": header_level, "title": header_title},
            )
        )

    return chunks
