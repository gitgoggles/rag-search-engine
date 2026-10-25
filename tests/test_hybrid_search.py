import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))

from lib.hybrid_search import (
    HybridSearch,
    correct_spelling,
    hybrid_score,
    normalize,
    normalize_command,
    rerank_batch,
    rerank_cross_encoder,
    rerank_individual,
    rrf_score,
    rrf_search_command,
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

    def test_pipeline_progress_messages(self):
        for method, args, fusion_message in [
            (self.search.rrf_search, ("bear", 60, 2),
             "[Search] Reciprocal rank fusion (RRF)..."),
            (self.search.weighted_search, ("bear", 0.25, 2),
             "[Search] Normalizing scores and weighted fusion..."),
        ]:
            with self.subTest(method=method.__name__):
                progress = io.StringIO()
                with contextlib.redirect_stderr(progress):
                    method(*args)
                self.assertEqual(progress.getvalue().splitlines(), [
                    "[Search] Keyword search (BM25)...",
                    "[Search] Semantic search...",
                    fusion_message,
                ])

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

    def test_rrf_merge_ranks_and_overfetch(self):
        results = self.search.rrf_search("bear", 60, 2)
        self.search._bm25_search.assert_called_once_with("bear", 1000)
        self.search.semantic_search.search_chunks.assert_called_once_with("bear", 1000)
        self.assertEqual([r["document"]["id"] for r in results], [1, 3, 2])
        self.assertEqual(results[0]["bm25_rank"], 1)
        self.assertEqual(results[0]["semantic_rank"], 2)
        self.assertAlmostEqual(results[0]["rrf_score"], 1 / 61 + 1 / 62)
        self.assertIsNone(results[1]["bm25_rank"])
        self.assertAlmostEqual(results[1]["rrf_score"], 1 / 61)
        self.assertIsNone(results[2]["semantic_rank"])
        self.assertEqual(rrf_score(1, 0), 1.0)

    def test_rrf_empty_sources_and_nonpositive_limit(self):
        self.assertEqual(self.search.rrf_search("bear", 60, 0), [])
        self.search._bm25_search.assert_not_called()
        self.search._bm25_search.return_value = []
        results = self.search.rrf_search("bear", 60)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["bm25_rank"] is None for r in results))
        self.search.semantic_search.search_chunks.return_value = []
        self.assertEqual(self.search.rrf_search("bear", 60), [])

    def test_rrf_command_format_and_limit(self):
        results = self.search.rrf_search("bear", 60, 2)
        output = io.StringIO()
        with (
            patch("lib.hybrid_search.load_movies", return_value=self.search.documents),
            patch("lib.hybrid_search.HybridSearch") as constructor,
            contextlib.redirect_stdout(output),
        ):
            constructor.return_value.rrf_search.return_value = results
            rrf_search_command("bear", 60, 1)
            constructor.return_value.rrf_search.assert_called_once_with("bear", 60, 1)
        self.assertEqual(
            output.getvalue(),
            "1. Movie 1\n  RRF Score: 0.033\n"
            "  BM25 Rank: 1, Semantic Rank: 2\n  A description\n\n",
        )

    def test_rrf_spell_enhancement(self):
        output = io.StringIO()
        with (
            patch("lib.hybrid_search.correct_spelling", return_value="british bear") as correct,
            patch("lib.hybrid_search.load_movies", return_value=[]),
            patch("lib.hybrid_search.HybridSearch") as constructor,
            contextlib.redirect_stdout(output),
        ):
            constructor.return_value.rrf_search.return_value = []
            rrf_search_command("briish bear", 60, 5, "spell")
            correct.assert_called_once_with("briish bear")
            constructor.return_value.rrf_search.assert_called_once_with("british bear", 60, 5)
        self.assertEqual(
            output.getvalue(),
            "Enhanced query (spell): 'briish bear' -> 'british bear'\n\n",
        )

    def test_rrf_debug_logs_pipeline_stages_and_cutoffs(self):
        results = self.search.rrf_search("bear", 60, 2)
        ranked = [
            {**result, "cross_encoder_score": float(rank)}
            for rank, result in enumerate(reversed(results))
        ]
        debug_output = io.StringIO()
        with (
            patch("lib.hybrid_search.correct_spelling", return_value="bear"),
            patch("lib.hybrid_search.load_movies", return_value=[]),
            patch("lib.hybrid_search.HybridSearch") as constructor,
            patch("lib.hybrid_search.rerank_cross_encoder", return_value=ranked),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(debug_output),
        ):
            constructor.return_value.rrf_search.return_value = results
            rrf_search_command("baer", 60, 1, "spell", "cross_encoder", debug=True)
        output = debug_output.getvalue()
        self.assertIn('[DEBUG] Original query: "baer"', output)
        self.assertIn('[DEBUG] Query after enhancements: "bear"', output)
        self.assertIn("Results after RRF search (before candidate cutoff)", output)
        self.assertIn("Re-ranking candidates", output)
        self.assertIn("Results after re-ranking (before output cutoff)", output)
        self.assertIn('"cross_encoder_score"', output)
        self.assertIn('"bm25_rank"', output)
        final_line = output.splitlines()[-1]
        self.assertIn("Final results", final_line)
        self.assertEqual(final_line.count('"title"'), 1)

    def test_spelling_response_and_empty_fallback(self):
        with (
            patch("lib.hybrid_search.load_dotenv"),
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
            patch("lib.hybrid_search.OpenAI") as client,
        ):
            completion = client.return_value.chat.completions.create
            for content, expected in [(" british bear\n", "british bear"), (None, "briish bear"), (" ", "briish bear")]:
                completion.return_value.choices = [Mock(message=Mock(content=content))]
                self.assertEqual(correct_spelling("briish bear"), expected)
            self.assertIn("briish bear", completion.call_args.kwargs["messages"][0]["content"])

    def test_individual_reranking(self):
        results = self.search.rrf_search("bear", 60, 2)
        with (
            patch("lib.hybrid_search.load_dotenv"),
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
            patch("lib.hybrid_search.OpenAI") as client,
            patch.dict("os.environ", {}, clear=True),
            patch("lib.hybrid_search.time.sleep") as sleep,
        ):
            completion = client.return_value.chat.completions.create
            completion.side_effect = [
                Mock(choices=[Mock(message=Mock(content=score))])
                for score in ["2", "9.5", "9.5"]
            ]
            ranked = rerank_individual("family bear movie", results)
            self.assertEqual([r["document"]["id"] for r in ranked], [3, 2, 1])
            self.assertEqual(ranked[0]["rerank_score"], 9.5)
            self.assertEqual(ranked[0]["rrf_score"], results[1]["rrf_score"])
            self.assertNotIn("rerank_score", results[0])
            self.assertEqual(sleep.call_count, 2)
            sleep.assert_called_with(3)
            client.assert_called_once_with(
                base_url="https://llama.rucknet.uk/v1", api_key="unused"
            )
            for call in completion.call_args_list:
                self.assertEqual(
                    call.kwargs["model"], "Qwen3.6-35B-A3B-UD-Q4_K_M"
                )
            self.assertEqual(completion.call_args.kwargs["max_tokens"], 32)
            self.assertEqual(completion.call_args.kwargs["extra_body"], {
                "chat_template_kwargs": {"enable_thinking": False}
            })
            prompt = completion.call_args_list[0].kwargs["messages"][0]["content"]
            self.assertIn("family bear movie", prompt)
            self.assertIn("A description", prompt)

    def test_rerank_provider_configuration(self):
        results = self.search.rrf_search("bear", 60, 1)[:1]
        with (
            patch("lib.hybrid_search.load_dotenv"),
            patch.dict("os.environ", {
                "RERANK_BASE_URL": "https://example.com/v1",
                "RERANK_MODEL": "custom-model",
                "RERANK_API_KEY": "custom-key",
            }, clear=True),
            patch("lib.hybrid_search.OpenAI") as client,
        ):
            completion = client.return_value.chat.completions.create
            completion.return_value = Mock(choices=[Mock(message=Mock(content="8"))])
            self.assertEqual(rerank_individual("bear", results)[0]["rerank_score"], 8)
            client.assert_called_once_with(
                base_url="https://example.com/v1", api_key="custom-key"
            )
            self.assertEqual(completion.call_args.kwargs["model"], "custom-model")

    def test_invalid_rerank_scores(self):
        results = self.search.rrf_search("bear", 60, 1)[:1]
        with (
            patch("lib.hybrid_search.load_dotenv"),
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
            patch("lib.hybrid_search.OpenAI") as client,
        ):
            for content in [None, "not a score", "User Safety: safe", "11", "-1", "nan", "inf"]:
                client.return_value.chat.completions.create.return_value = Mock(
                    choices=[Mock(message=Mock(content=content))]
                )
                with self.assertRaises(ValueError):
                    rerank_individual("bear", results)
        self.assertEqual(rerank_individual("bear", []), [])

    def test_rrf_rerank_candidate_limit_and_output(self):
        results = self.search.rrf_search("bear", 60, 2)
        candidates = results * 4
        ranked = [{**results[1], "rerank_score": 10.0},
                  {**results[0], "rerank_score": 2.0}]
        output = io.StringIO()
        with (
            patch("lib.hybrid_search.load_movies", return_value=[]),
            patch("lib.hybrid_search.HybridSearch") as constructor,
            patch("lib.hybrid_search.rerank_individual", return_value=ranked) as rerank,
            contextlib.redirect_stdout(output),
        ):
            constructor.return_value.rrf_search.return_value = candidates
            rrf_search_command("bear", 60, 1, rerank_method="individual")
            constructor.return_value.rrf_search.assert_called_once_with("bear", 60, 5)
            rerank.assert_called_once_with("bear", candidates[:5])
        self.assertIn("1. Movie 3", output.getvalue())
        self.assertIn("Re-rank Score: 10.000/10", output.getvalue())
        self.assertIn("RRF Score: 0.016", output.getvalue())
        self.assertNotIn("2. Movie", output.getvalue())

    def test_batch_reranking(self):
        results = self.search.rrf_search("bear", 60, 2)
        with (
            patch("lib.hybrid_search.load_dotenv"),
            patch.dict("os.environ", {
                "RERANK_BASE_URL": "https://example.com/v1",
                "RERANK_MODEL": "custom-model",
                "RERANK_API_KEY": "custom-key",
            }, clear=True),
            patch("lib.hybrid_search.OpenAI") as client,
        ):
            completion = client.return_value.chat.completions.create
            completion.return_value = Mock(choices=[Mock(message=Mock(content="[2, 1, 3]"))])
            ranked = rerank_batch("family bear movie", results)
            completion.assert_called_once()
            client.assert_called_once_with(
                base_url="https://example.com/v1", api_key="custom-key"
            )
            self.assertEqual(completion.call_args.kwargs["model"], "custom-model")
            prompt = completion.call_args.kwargs["messages"][0]["content"]
            self.assertIn("family bear movie", prompt)
            for result in results:
                self.assertIn(result["document"]["title"], prompt)
                self.assertIn(result["document"]["description"], prompt)
            self.assertEqual([r["document"]["id"] for r in ranked], [2, 1, 3])
            self.assertEqual([r["rerank_rank"] for r in ranked], [1, 2, 3])
            self.assertEqual(ranked[0]["rrf_score"], results[2]["rrf_score"])
            self.assertNotIn("rerank_rank", results[0])

    def test_batch_invalid_responses_and_empty_results(self):
        results = self.search.rrf_search("bear", 60, 2)
        with (
            patch("lib.hybrid_search.load_dotenv"),
            patch("lib.hybrid_search.OpenAI") as client,
        ):
            self.assertEqual(rerank_batch("bear", []), [])
            client.assert_not_called()
            for content in [None, "not JSON", "{}", "[1, 2]", "[1, 1, 3]",
                            "[1, 2, 4]", '["1", 2, 3]', "[true, 2, 3]",
                            "[1.0, 2, 3]", "[[1], 2, 3]"]:
                with self.subTest(content=content):
                    client.return_value.chat.completions.create.return_value = Mock(
                        choices=[Mock(message=Mock(content=content))]
                    )
                    with self.assertRaises(ValueError):
                        rerank_batch("bear", results)

    def test_rrf_batch_candidate_limit_and_output(self):
        results = self.search.rrf_search("bear", 60, 2)
        candidates = results * 4
        ranked = [{**results[1], "rerank_rank": 1},
                  {**results[0], "rerank_rank": 2}]
        output = io.StringIO()
        with (
            patch("lib.hybrid_search.load_movies", return_value=[]),
            patch("lib.hybrid_search.HybridSearch") as constructor,
            patch("lib.hybrid_search.rerank_batch", return_value=ranked) as rerank,
            contextlib.redirect_stdout(output),
        ):
            constructor.return_value.rrf_search.return_value = candidates
            rrf_search_command("bear", 60, 1, rerank_method="batch")
            constructor.return_value.rrf_search.assert_called_once_with("bear", 60, 5)
            rerank.assert_called_once_with("bear", candidates[:5])
        self.assertIn("Re-ranking top 1 results using batch method...", output.getvalue())
        self.assertIn("1. Movie 3\n  Re-rank Rank: 1\n  RRF Score: 0.016", output.getvalue())
        self.assertNotIn("2. Movie", output.getvalue())

    def test_cross_encoder_reranking(self):
        results = self.search.rrf_search("bear", 60, 2)
        with patch("lib.hybrid_search.CrossEncoder") as encoder:
            encoder.return_value.predict.return_value = [-2.0, 3.314, -1.0]
            ranked = rerank_cross_encoder("family bear movie", results)
            encoder.assert_called_once_with(
                "cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu"
            )
            encoder.return_value.predict.assert_called_once_with([
                ["family bear movie", f"{r['document']['title']} - A description"]
                for r in results
            ])
        self.assertEqual([r["document"]["id"] for r in ranked], [3, 2, 1])
        self.assertEqual(ranked[0]["cross_encoder_score"], 3.314)
        self.assertEqual(ranked[0]["rrf_score"], results[1]["rrf_score"])
        self.assertNotIn("cross_encoder_score", results[0])

    def test_cross_encoder_empty_results_and_missing_fields(self):
        with patch("lib.hybrid_search.CrossEncoder") as encoder:
            self.assertEqual(rerank_cross_encoder("bear", []), [])
            encoder.assert_not_called()
            encoder.return_value.predict.return_value = [0.0]
            rerank_cross_encoder("bear", [{"document": {}}])
            encoder.return_value.predict.assert_called_once_with([["bear", " - "]])

    def test_rrf_cross_encoder_candidate_limit_and_output(self):
        results = self.search.rrf_search("bear", 60, 2)
        candidates = results * 4
        ranked = [{**results[1], "cross_encoder_score": 3.314},
                  {**results[0], "cross_encoder_score": -1.0}]
        output = io.StringIO()
        with (
            patch("lib.hybrid_search.load_movies", return_value=[]),
            patch("lib.hybrid_search.HybridSearch") as constructor,
            patch("lib.hybrid_search.rerank_cross_encoder", return_value=ranked) as rerank,
            contextlib.redirect_stdout(output),
        ):
            constructor.return_value.rrf_search.return_value = candidates
            rrf_search_command("bear", 60, 1, rerank_method="cross_encoder")
            constructor.return_value.rrf_search.assert_called_once_with("bear", 60, 5)
            rerank.assert_called_once_with("bear", candidates[:5])
        self.assertIn("Re-ranking top 1 results using cross_encoder method...", output.getvalue())
        self.assertIn("1. Movie 3\n  Cross Encoder Score: 3.314\n  RRF Score: 0.016", output.getvalue())
        self.assertNotIn("2. Movie", output.getvalue())

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
