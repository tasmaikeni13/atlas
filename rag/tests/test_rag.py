"""Automated integration and regression tests for ATLAS Research RAG."""

from __future__ import annotations

import unittest
from pathlib import Path

from rag import index_repository, search
from rag.config import DB_PATH, REPO_ROOT
from rag.store.db import RAGDatabase


class TestATLASRAG(unittest.TestCase):
    """Test suite for RAG indexing, categorization, and retrieval precision."""

    @classmethod
    def setUpClass(cls):
        # Ensure database is indexed
        cls.indexed_count = index_repository(db_path=DB_PATH, repo_root=REPO_ROOT, verbose=False)

    def test_indexing_count(self):
        self.assertGreater(self.indexed_count, 100, "Should index at least 100 research chunks")
        db = RAGDatabase(DB_PATH)
        stats = db.get_stats()
        self.assertIn("code", stats)
        self.assertIn("theory_proof", stats)
        self.assertIn("theory_paper", stats)
        self.assertIn("runs", stats)

    def test_lean_proof_retrieval(self):
        results = search("alloc_lower_bound", category="proof", top_k=3)
        self.assertTrue(len(results) > 0, "Should retrieve alloc_lower_bound theorem")
        top_hit = results[0]
        self.assertIn("alloc_lower_bound", top_hit["symbol"])
        self.assertEqual(top_hit["category"], "theory_proof")
        self.assertIn("Certificates.lean", top_hit["rel_path"])
        self.assertGreaterEqual(top_hit["start_line"], 1)

    def test_paper_theory_retrieval(self):
        results = search("Continuous Minimax Budget Allocation", category="paper", top_k=3)
        self.assertTrue(len(results) > 0, "Should retrieve paper section on budget allocation")
        top_hit = results[0]
        self.assertIn("paper/atlas.tex", top_hit["rel_path"])
        self.assertIn("Minimax Budget Allocation", top_hit["title"])

    def test_code_ast_retrieval(self):
        results = search("HermiteTaylorReconstruction evaluate_batch", category="code", top_k=3)
        self.assertTrue(len(results) > 0, "Should retrieve JAX HermiteTaylorReconstruction method")
        top_hit = results[0]
        self.assertIn("reconstruct.py", top_hit["rel_path"])
        self.assertIn("HermiteTaylorReconstruction", top_hit["symbol"])

    def test_runs_metrics_retrieval(self):
        results = search("inflation_factors true_sharpness", category="runs", top_k=3)
        self.assertTrue(len(results) > 0, "Should retrieve sharpness audit JSON")
        top_hit = results[0]
        self.assertIn("runs/sharpness", top_hit["rel_path"])
        self.assertIn("inflation_factors", top_hit["content"])

    def test_skill_protocol_retrieval(self):
        results = search("falsifiable ML result research contract", category="skill", top_k=3)
        self.assertTrue(len(results) > 0, "Should retrieve ml-research skill")
        top_hit = results[0]
        self.assertIn("skills/", top_hit["rel_path"])


if __name__ == "__main__":
    unittest.main()
