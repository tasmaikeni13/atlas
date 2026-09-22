"""AST-aware Python code parser that extracts classes, functions, and docstrings."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

from . import DocumentChunk


def parse_python_file(file_path: Path, repo_root: Path, category: str = "code") -> List[DocumentChunk]:
    """Parse a Python source file into AST-based semantic chunks preserving context and lines."""
    try:
        source_code = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return []

    rel_path = str(file_path.relative_to(repo_root))
    lines = source_code.splitlines()
    total_lines = len(lines)
    chunks: List[DocumentChunk] = []

    try:
        tree = ast.parse(source_code, filename=str(file_path))
    except Exception:
        # Fallback to line-based chunking if AST parsing fails
        return _fallback_chunking(file_path, rel_path, lines, category)

    # 1. Module-level docstring and overview chunk
    module_doc = ast.get_docstring(tree)
    if module_doc:
        doc_lines = module_doc.strip().splitlines()
        chunks.append(
            DocumentChunk(
                file_path=str(file_path),
                rel_path=rel_path,
                category=category,
                title=f"{rel_path} :: Module Overview",
                symbol=file_path.stem,
                start_line=1,
                end_line=min(len(doc_lines) + 5, total_lines),
                content=f"Module: {rel_path}\nDescription:\n{module_doc}",
                metadata={"type": "module_doc"},
            )
        )

    # 2. Iterate top-level AST nodes
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            chunks.extend(_extract_class_chunks(node, lines, file_path, rel_path, category))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(_extract_function_chunk(node, lines, file_path, rel_path, category))
        elif isinstance(node, ast.If) and getattr(node.test, "id", None) == "__name__":
            # Main execution block
            start_l = node.lineno
            end_l = getattr(node, "end_lineno", len(lines))
            block_content = "\n".join(lines[start_l - 1 : end_l])
            chunks.append(
                DocumentChunk(
                    file_path=str(file_path),
                    rel_path=rel_path,
                    category=category,
                    title=f"{rel_path} :: __main__ entrypoint",
                    symbol="__main__",
                    start_line=start_l,
                    end_line=end_l,
                    content=block_content,
                    metadata={"type": "main_block"},
                )
            )

    # If the file has no classes or functions (e.g., simple script or __init__.py), capture overall content
    if not chunks:
        chunks.append(
            DocumentChunk(
                file_path=str(file_path),
                rel_path=rel_path,
                category=category,
                title=f"{rel_path} :: Script",
                symbol=file_path.stem,
                start_line=1,
                end_line=total_lines,
                content=source_code[:3000],
                metadata={"type": "script"},
            )
        )

    return chunks


def _extract_function_chunk(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: List[str],
    file_path: Path,
    rel_path: str,
    category: str,
    parent_class: str = "",
) -> DocumentChunk:
    start_l = node.lineno
    end_l = getattr(node, "end_lineno", start_l)
    code_slice = "\n".join(lines[start_l - 1 : end_l])
    docstring = ast.get_docstring(node) or ""

    symbol = f"{parent_class}.{node.name}" if parent_class else node.name
    title = f"{rel_path} :: {symbol}"

    return DocumentChunk(
        file_path=str(file_path),
        rel_path=rel_path,
        category=category,
        title=title,
        symbol=symbol,
        start_line=start_l,
        end_line=end_l,
        content=f"# Symbol: {symbol}\n# Docstring: {docstring}\n{code_slice}",
        metadata={
            "type": "function",
            "name": node.name,
            "parent_class": parent_class,
            "has_docstring": bool(docstring),
        },
    )


def _extract_class_chunks(
    node: ast.ClassDef,
    lines: List[str],
    file_path: Path,
    rel_path: str,
    category: str,
) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    class_start = node.lineno
    class_end = getattr(node, "end_lineno", class_start)
    class_doc = ast.get_docstring(node) or ""

    # Class summary chunk
    class_header_lines: List[str] = []
    for i in range(class_start - 1, min(class_start + 15, len(lines))):
        class_header_lines.append(lines[i])

    chunks.append(
        DocumentChunk(
            file_path=str(file_path),
            rel_path=rel_path,
            category=category,
            title=f"{rel_path} :: class {node.name}",
            symbol=node.name,
            start_line=class_start,
            end_line=class_end,
            content=f"class {node.name}:\n'''{class_doc}'''\n" + "\n".join(class_header_lines),
            metadata={"type": "class", "name": node.name, "doc": class_doc},
        )
    )

    # Class methods
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(
                _extract_function_chunk(
                    item,
                    lines,
                    file_path,
                    rel_path,
                    category,
                    parent_class=node.name,
                )
            )

    return chunks


def _fallback_chunking(
    file_path: Path,
    rel_path: str,
    lines: List[str],
    category: str,
    window_size: int = 60,
    overlap: int = 15,
) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    total = len(lines)
    step = max(1, window_size - overlap)

    for i in range(0, total, step):
        start = i + 1
        end = min(total, i + window_size)
        content = "\n".join(lines[i:end])
        chunks.append(
            DocumentChunk(
                file_path=str(file_path),
                rel_path=rel_path,
                category=category,
                title=f"{rel_path}#L{start}-L{end}",
                symbol=file_path.stem,
                start_line=start,
                end_line=end,
                content=content,
                metadata={"type": "fallback_chunk"},
            )
        )
        if end >= total:
            break

    return chunks
