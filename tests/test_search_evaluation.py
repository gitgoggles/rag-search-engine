import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))

import hybrid_search_cli
from lib.hybrid_search import evaluate_results, rrf_search_command


class SearchEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            {"document": {"id": i, "title": f"Movie {i}", "description": "Full plot"},
             "rrf_score": 0.03, "bm25_rank": i, "semantic_rank": i}
            for i in range(1, 4)
        ]

    def test_scores_and_prompt(self):
        with (
            patch("lib.hybrid_search.load_dotenv"),
            patch("lib.hybrid_search.OpenAI") as client,
        ):
            completion = client.return_value.chat.completions.create
            completion.return_value = Mock(choices=[Mock(message=Mock(content="[3, 0, 2]"))])
            self.assertEqual(evaluate_results("bear family", self.results), [3, 0, 2])
            prompt = completion.call_args.kwargs["messages"][0]["content"]
            self.assertIn("bear family", prompt)
            for result in self.results:
                self.assertIn(result["document"]["title"], prompt)
            self.assertIn("Full plot", prompt)

    def test_invalid_responses_and_empty_results(self):
        with (
            patch("lib.hybrid_search.load_dotenv"),
            patch("lib.hybrid_search.OpenAI") as client,
        ):
            self.assertEqual(evaluate_results("bear", []), [])
            client.assert_not_called()
            for content in [None, "not JSON", "{}", "[3]", "[0, 1, 2, 3]",
                            "[true, 1, 2]", "[1.0, 1, 2]", '["3", 1, 2]',
                            "[-1, 1, 2]", "[4, 1, 2]", "[[], 1, 2]"]:
                with self.subTest(content=content):
                    client.return_value.chat.completions.create.return_value = Mock(
                        choices=[Mock(message=Mock(content=content))]
                    )
                    with self.assertRaises(ValueError):
                        evaluate_results("bear", self.results)

    def test_report_after_search_uses_final_order_and_original_query(self):
        output = io.StringIO()
        ranked = [{**result, "rerank_rank": rank}
                  for rank, result in enumerate(reversed(self.results), start=1)]
        with (
            patch("lib.hybrid_search.load_movies", return_value=[]),
            patch("lib.hybrid_search.HybridSearch") as search,
            patch("lib.hybrid_search.rewrite_query", return_value="enhanced bear"),
            patch("lib.hybrid_search.rerank_batch", return_value=ranked),
            patch("lib.hybrid_search.evaluate_results", return_value=[3, 1]) as evaluate,
            contextlib.redirect_stdout(output),
        ):
            search.return_value.rrf_search.return_value = self.results
            rrf_search_command("bear", 60, 2, enhance="rewrite",
                               rerank_method="batch", evaluate=True)
        evaluate.assert_called_once_with("bear", ranked[:2])
        self.assertTrue(output.getvalue().endswith("1. Movie 3: 3/3\n2. Movie 2: 1/3\n"))
        self.assertLess(output.getvalue().index("RRF Score:"), output.getvalue().index("3/3"))

    def test_evaluation_disabled_by_default(self):
        with (
            patch("lib.hybrid_search.load_movies", return_value=[]),
            patch("lib.hybrid_search.HybridSearch") as search,
            patch("lib.hybrid_search.evaluate_results") as evaluate,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            search.return_value.rrf_search.return_value = []
            rrf_search_command("bear", 60, 5)
        evaluate.assert_not_called()

    def test_cli_flag(self):
        for enabled in [False, True]:
            with (
                patch.object(sys, "argv", ["hybrid_search_cli.py", "rrf-search", "bear"]
                             + (["--evaluate"] if enabled else [])),
                patch.object(hybrid_search_cli.hs, "rrf_search_command") as command,
            ):
                hybrid_search_cli.main()
            self.assertEqual(command.call_args.kwargs["evaluate"], enabled)


if __name__ == "__main__":
    unittest.main()
