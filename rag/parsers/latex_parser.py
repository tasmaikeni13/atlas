"""LaTeX and BibTeX parser for academic papers, theorems, and proofs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from . import DocumentChunk

SECTION_SPLIT_REGEX = re.compile(
    r"(?P<marker>\\(?:section|subsection|subsubsection|begin\{abstract\}|begin\{theorem\}|begin\{lemma\}|begin\{proposition\}|begin\{definition\}))\*?\{?(?P<title>[^}\n]*)\}?",
    re.MULTILINE,
)


def parse_latex_file(file_path: Path, repo_root: Path, category: str = "theory_paper") -> List[DocumentChunk]:
    """Parse a LaTeX paper (.tex) or bibliography (.bib) file into section-aware chunks."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    rel_path = str(file_path.relative_to(repo_root))
    lines = content.splitlines()
    total_lines = len(lines)
    chunks: List[DocumentChunk] = []

    if file_path.suffix == ".bib":
        return _parse_bib_file(file_path, rel_path, lines, category)

    # Find all section/abstract/theorem markers
    matches = list(SECTION_SPLIT_REGEX.finditer(content))

    if not matches:
        chunks.append(
            DocumentChunk(
                file_path=str(file_path),
                rel_path=rel_path,
                category=category,
                title=f"{rel_path} :: Full Paper Text",
                symbol=file_path.stem,
                start_line=1,
                end_line=total_lines,
                content=content[:3500],
                metadata={"type": "tex_doc"},
            )
        )
        return chunks

    # Add introduction / title preamble if before first section
    first_start_char = matches[0].start()
    first_line = content[:first_start_char].count("\n") + 1
    if first_line > 10:
        preamble_text = "\n".join(lines[: first_line - 1]).strip()
        # Clean title if present
        title_match = re.search(r"\\title\{([\s\S]*?)\}", preamble_text)
        paper_title = title_match.group(1).replace("\\\\", " ").strip() if title_match else "Preamble"
        chunks.append(
            DocumentChunk(
                file_path=str(file_path),
                rel_path=rel_path,
                category=category,
                title=f"{rel_path} :: {paper_title}",
                symbol="Title",
                start_line=1,
                end_line=first_line - 1,
                content=_clean_tex_for_search(preamble_text),
                metadata={"type": "tex_preamble", "title": paper_title},
            )
        )

    for idx, match in enumerate(matches):
        marker = match.group("marker")
        title_raw = match.group("title").strip() or marker.replace("\\", "")

        start_char = match.start()
        start_line = content[:start_char].count("\n") + 1

        if idx + 1 < len(matches):
            next_start_char = matches[idx + 1].start()
            end_line = content[:next_start_char].count("\n")
        else:
            end_line = total_lines

        chunk_lines = lines[start_line - 1 : end_line]
        chunk_text = "\n".join(chunk_lines).strip()
        cleaned_text = _clean_tex_for_search(chunk_text)

        chunks.append(
            DocumentChunk(
                file_path=str(file_path),
                rel_path=rel_path,
                category=category,
                title=f"{rel_path} :: {title_raw}",
                symbol=title_raw,
                start_line=start_line,
                end_line=end_line,
                content=f"Paper Section: {title_raw}\n\n{cleaned_text}",
                metadata={"type": "tex_section", "marker": marker, "title": title_raw},
            )
        )

    return chunks


def _parse_bib_file(file_path: Path, rel_path: str, lines: List[str], category: str) -> List[DocumentChunk]:
    """Parse BibTeX entries."""
    chunks: List[DocumentChunk] = []
    text = "\n".join(lines)
    entries = list(re.finditer(r"@([a-zA-Z]+)\{([^,\s]+),([\s\S]*?)\n\}", text))

    for entry in entries:
        kind = entry.group(1)
        cite_key = entry.group(2)
        body = entry.group(3)
        start_line = text[: entry.start()].count("\n") + 1
        end_line = text[: entry.end()].count("\n") + 1

        chunks.append(
            DocumentChunk(
                file_path=str(file_path),
                rel_path=rel_path,
                category=category,
                title=f"{rel_path} :: @{kind}[{cite_key}]",
                symbol=cite_key,
                start_line=start_line,
                end_line=end_line,
                content=f"Citation: [{cite_key}] ({kind})\n{body.strip()}",
                metadata={"type": "bib_entry", "cite_key": cite_key, "kind": kind},
            )
        )
    return chunks


def _clean_tex_for_search(text: str) -> str:
    """Lightly clean LaTeX formatting while retaining mathematical formulas and semantics."""
    # Remove comments
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)
    # Normalize common macros
    text = re.sub(r"\\textbf\{([^}]+)\}", r"**\1**", text)
    text = re.sub(r"\\textit\{([^}]+)\}", r"*\1*", text)
    text = re.sub(r"\\emph\{([^}]+)\}", r"*\1*", text)
    text = re.sub(r"\\cite\{([^}]+)\}", r"[\1]", text)
    text = re.sub(r"\\ref\{([^}]+)\}", r"Ref(\1)", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
