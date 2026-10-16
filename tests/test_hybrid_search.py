import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))

from lib.hybrid_search import (
    HybridSearch,
    hybrid_score,
    normalize,
    normalize_command,
    weighted_search_command,
)


class HybridSearchTests(unittest.TestCase):
    def setUp(self):
        # Avoid loading a model or touching caches in unit tests.
        self.search = HybridSearch.__new__(HybridSearch)
        self.search.documents = [
            {"id": i, "title": f"Movie {i}", "description": "A description"}
            for i in range(1, 4)
        ]
        self.search._bm25_search = Mock(return_value=[(1, 10.0), (2, 5.0)])
        self.search.semantic_search = Mock()
        self.search.semantic_search.search_chunks.return_value = [
            {"id": 3, "score": 0.9},
            {"id": 1, "score": 0.3},
        ]

    def test_normalize(self):
        self.assertEqual(normalize([]), [])
        self.assertEqual(normalize([4.0]), [1.0])
        self.assertEqual(normalize([4.0, 4.0]), [1.0, 1.0])
        self.assertEqual(normalize([-2.0, 0.0, 2.0]), [0.0, 0.5, 1.0])

    def test_normalize_command_format(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            normalize_command([2.0, 4.0])
        self.assertEqual(output.getvalue(), "* 0.0000\n* 1.0000\n")

    def test_merge_by_id_and_overfetch(self):
        results = self.search.weighted_search("bear", 0.25, 2)
        self.search._bm25_search.assert_called_once_with("bear", 1000)
        self.search.semantic_search.search_chunks.assert_called_once_with("bear", 1000)
        self.assertEqual([r["document"]["id"] for r in results], [3, 1, 2])
        self.assertEqual([r["hybrid_score"] for r in results], [0.75, 0.25, 0.0])
        self.assertEqual(results[0]["bm25_score"], 0.0)
        self.assertEqual(results[1]["semantic_score"], 0.0)

    def test_alpha_endpoints(self):
        self.assertEqual(hybrid_score(0.8, 0.2, 1.0), 0.8)
        self.assertEqual(hybrid_score(0.8, 0.2, 0.0), 0.2)

    def test_empty_sources(self):
        self.search._bm25_search.return_value = []
        results = self.search.weighted_search("bear", 0.5)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["bm25_score"] == 0.0 for r in results))
        self.search.semantic_search.search_chunks.return_value = []
        self.assertEqual(self.search.weighted_search("bear", 0.5), [])

    def test_invalid_alpha_and_nonpositive_limit(self):
        for alpha in [-0.1, 1.1]:
            with self.assertRaises(ValueError):
                self.search.weighted_search("bear", alpha)
        self.assertEqual(self.search.weighted_search("bear", 0.5, 0), [])
        self.search._bm25_search.assert_not_called()

    def test_command_formats_and_limits_output(self):
        results = self.search.weighted_search("bear", 0.25, 2)
        output = io.StringIO()
        with (
            patch("lib.hybrid_search.load_movies", return_value=self.search.documents),
            patch("lib.hybrid_search.HybridSearch") as constructor,
            contextlib.redirect_stdout(output),
        ):
            constructor.return_value.weighted_search.return_value = results
            weighted_search_command("bear", 0.25, 1)
            constructor.return_value.weighted_search.assert_called_once_with("bear", 0.25, 1)
        self.assertEqual(
            output.getvalue(),
            "1. Movie 3\nHybrid Score: 0.750\nBM25: 0.000, Semantic: 1.000\nA description\n",
        )


if __name__ == "__main__":
    unittest.main()
