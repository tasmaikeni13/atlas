"""SQLite FTS5 storage and retrieval engine for ATLAS Research RAG."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.parsers import DocumentChunk
from rag.store.rerank import rerank_results


class RAGDatabase:
    """Manages the full-text indexed SQLite database for fast semantic and symbol retrieval."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create documents table and FTS5 virtual table with sync triggers."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    rel_path TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    symbol TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    content TEXT NOT NULL,
                    metadata TEXT
                );
                """
            )

            # Check if virtual table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fts_documents';")
            if not cursor.fetchone():
                cursor.execute(
                    """
                    CREATE VIRTUAL TABLE fts_documents USING fts5(
                        title,
                        symbol,
                        content,
                        rel_path,
                        category,
                        content='documents',
                        content_rowid='id',
                        tokenize='porter unicode61'
                    );
                    """
                )

                # Triggers to keep FTS in sync with primary table
                cursor.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS trg_docs_ai AFTER INSERT ON documents BEGIN
                        INSERT INTO fts_documents(rowid, title, symbol, content, rel_path, category)
                        VALUES (new.id, new.title, new.symbol, new.content, new.rel_path, new.category);
                    END;
                    """
                )
                cursor.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS trg_docs_ad AFTER DELETE ON documents BEGIN
                        INSERT INTO fts_documents(fts_documents, rowid, title, symbol, content, rel_path, category)
                        VALUES ('delete', old.id, old.title, old.symbol, old.content, old.rel_path, old.category);
                    END;
                    """
                )
                cursor.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS trg_docs_au AFTER UPDATE ON documents BEGIN
                        INSERT INTO fts_documents(fts_documents, rowid, title, symbol, content, rel_path, category)
                        VALUES ('delete', old.id, old.title, old.symbol, old.content, old.rel_path, old.category);
                        INSERT INTO fts_documents(rowid, title, symbol, content, rel_path, category)
                        VALUES (new.id, new.title, new.symbol, new.content, new.rel_path, new.category);
                    END;
                    """
                )
            conn.commit()

    def clear(self) -> None:
        """Clear all indexed documents."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM documents;")
            conn.execute("INSERT INTO fts_documents(fts_documents) VALUES('rebuild');")
            conn.commit()

    def insert_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Bulk insert parsed document chunks."""
        if not chunks:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            rows = [
                (
                    c.file_path,
                    c.rel_path,
                    c.category,
                    c.title,
                    c.symbol,
                    c.start_line,
                    c.end_line,
                    c.content,
                    json.dumps(c.metadata),
                )
                for c in chunks
            ]
            cursor.executemany(
                """
                INSERT INTO documents (file_path, rel_path, category, title, symbol, start_line, end_line, content, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
            conn.commit()
            return len(rows)

    def count(self) -> int:
        """Get total number of indexed chunks."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT count(*) as total FROM documents;").fetchone()
            return int(row["total"]) if row else 0

    def get_stats(self) -> Dict[str, int]:
        """Get count of chunks per category."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT category, count(*) as cnt FROM documents GROUP BY category;").fetchall()
            return {r["category"]: r["cnt"] for r in rows}

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5,
        candidate_pool: int = 30,
    ) -> List[Dict[str, Any]]:
        """Search indexed chunks using FTS5 BM25 with query expansion and smart reranking."""
        query_clean = query.strip()
        if not query_clean:
            return []

        # Tokenize query words
        tokens = [re.sub(r"[^\w]", "", t) for t in query_clean.split()]
        tokens = [t for t in tokens if len(t) > 0]
        if not tokens:
            return []

        # Build FTS5 expression: prioritize prefix match and OR expansion
        fts_terms = []
        for t in tokens:
            fts_terms.append(f'"{t}"*')
        fts_query = " OR ".join(fts_terms)

        results: List[Dict[str, Any]] = []

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Primary FTS5 query
            params: List[Any] = [fts_query]
            category_filter = ""
            if category:
                category_filter = "AND d.category LIKE ?"
                params.append(f"%{category}%")

            params.append(candidate_pool)

            sql = f"""
                SELECT d.id, d.file_path, d.rel_path, d.category, d.title, d.symbol,
                       d.start_line, d.end_line, d.content, d.metadata,
                       bm25(fts_documents, 2.0, 3.0, 1.0, 1.5, 0.5) as rank
                FROM documents d
                JOIN fts_documents f ON d.id = f.rowid
                WHERE fts_documents MATCH ? {category_filter}
                ORDER BY rank
                LIMIT ?;
            """

            try:
                rows = cursor.execute(sql, params).fetchall()
                for r in rows:
                    results.append(
                        {
                            "id": r["id"],
                            "file_path": r["file_path"],
                            "rel_path": r["rel_path"],
                            "category": r["category"],
                            "title": r["title"],
                            "symbol": r["symbol"],
                            "start_line": r["start_line"],
                            "end_line": r["end_line"],
                            "content": r["content"],
                            "metadata": json.loads(r["metadata"] or "{}"),
                            "rank": r["rank"],
                        }
                    )
            except sqlite3.OperationalError:
                # In case of syntax error in FTS tokenization, fallback to simple LIKE
                pass

            # 2. Fallback if FTS produced 0 results
            if not results:
                like_clauses = []
                like_params: List[Any] = []
                for t in tokens[:3]:
                    like_clauses.append("(d.content LIKE ? OR d.title LIKE ? OR d.symbol LIKE ?)")
                    like_params.extend([f"%{t}%", f"%{t}%", f"%{t}%"])
                
                where_clause = " OR ".join(like_clauses)
                if category:
                    where_clause = f"({where_clause}) AND d.category LIKE ?"
                    like_params.append(f"%{category}%")

                like_params.append(candidate_pool)
                fallback_sql = f"""
                    SELECT d.id, d.file_path, d.rel_path, d.category, d.title, d.symbol,
                           d.start_line, d.end_line, d.content, d.metadata, -1.0 as rank
                    FROM documents d
                    WHERE {where_clause}
                    LIMIT ?;
                """
                try:
                    fb_rows = cursor.execute(fallback_sql, like_params).fetchall()
                    for r in fb_rows:
                        results.append(
                            {
                                "id": r["id"],
                                "file_path": r["file_path"],
                                "rel_path": r["rel_path"],
                                "category": r["category"],
                                "title": r["title"],
                                "symbol": r["symbol"],
                                "start_line": r["start_line"],
                                "end_line": r["end_line"],
                                "content": r["content"],
                                "metadata": json.loads(r["metadata"] or "{}"),
                                "rank": r["rank"],
                            }
                        )
                except Exception:
                    pass

        # Apply intelligent lexical and symbol reranking
        return rerank_results(query_clean, results, top_k=top_k)
