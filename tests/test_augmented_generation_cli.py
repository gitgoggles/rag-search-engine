import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))

import augmented_generation_cli as cli


class AugmentedGenerationCliTests(unittest.TestCase):
    def test_rag_uses_top_five_documents_and_prints_answer(self):
        documents = [
            {"title": f"Movie {i}", "description": f"Description {i}"}
            for i in range(6)
        ]
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["augmented_generation_cli.py", "rag", "dinosaurs"]),
            patch.object(cli, "load_dotenv"),
            patch.dict(cli.os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            patch.object(cli, "load_movies", return_value=documents),
            patch.object(cli, "HybridSearch") as search,
            patch.object(cli, "OpenAI") as client,
            contextlib.redirect_stdout(output),
        ):
            search.return_value.rrf_search.return_value = [
                {"document": doc} for doc in documents
            ]
            completions = client.return_value.chat.completions
            completions.create.return_value.choices[0].message.content = "Try Movie 0."
            cli.main()

        search.assert_called_once_with(documents)
        search.return_value.rrf_search.assert_called_once_with("dinosaurs", k=60, limit=5)
        prompt = completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("Query: dinosaurs", prompt)
        for doc in documents[:5]:
            self.assertIn(doc["title"], prompt)
            self.assertIn(doc["description"], prompt)
        self.assertNotIn("Movie 5", prompt)
        self.assertEqual(
            output.getvalue(),
            "Search Results:\n" + "".join(f"- Movie {i}\n" for i in range(5))
            + "\nRAG Response:\nTry Movie 0.\n",
        )

    def test_summarize_limits_results_and_prints_summary(self):
        documents = [
            {"title": f"Movie {i}", "description": f"Description {i}"}
            for i in range(6)
        ]
        for limit in (5, 2):
            with self.subTest(limit=limit):
                output = io.StringIO()
                progress = io.StringIO()
                argv = ["augmented_generation_cli.py", "summarize", "dinosaurs"]
                if limit != 5:
                    argv.extend(["--limit", str(limit)])
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(cli, "load_dotenv"),
                    patch.dict(cli.os.environ, {"OPENROUTER_API_KEY": "test-key"}),
                    patch.object(cli, "load_movies", return_value=documents),
                    patch.object(cli, "HybridSearch") as search,
                    patch.object(cli, "OpenAI") as client,
                    contextlib.redirect_stdout(output),
                    contextlib.redirect_stderr(progress),
                ):
                    search.return_value.rrf_search.return_value = [
                        {"document": doc} for doc in documents
                    ]
                    completions = client.return_value.chat.completions
                    completions.create.return_value.choices[0].message.content = "  Summary.  "
                    def respond(**kwargs):
                        self.assertEqual(
                            progress.getvalue(),
                            "[LLM] Prompting for search result summarization...\n",
                        )
                        return completions.create.return_value

                    completions.create.side_effect = respond
                    cli.main()

                search.assert_called_once_with(documents)
                search.return_value.rrf_search.assert_called_once_with(
                    "dinosaurs", k=60, limit=limit
                )
                prompt = completions.create.call_args.kwargs["messages"][0]["content"]
                self.assertIn("Query: dinosaurs", prompt)
                self.assertIn("Webflyx", prompt)
                self.assertIn("3-4 sentence", prompt)
                for doc in documents[:limit]:
                    self.assertIn(doc["title"], prompt)
                    self.assertIn(doc["description"], prompt)
                self.assertNotIn(f"Movie {limit}", prompt)
                self.assertEqual(
                    output.getvalue(),
                    "Search Results:\n"
                    + "".join(f"- Movie {i}\n" for i in range(limit))
                    + "\nLLM Summary:\nSummary.\n",
                )

    def test_summarize_requires_query_and_positive_limit(self):
        for args in (["summarize"], ["summarize", "dinosaurs", "--limit", "0"],
                     ["summarize", "dinosaurs", "--limit", "-1"]):
            with (
                self.subTest(args=args),
                patch.object(sys, "argv", ["augmented_generation_cli.py", *args]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                cli.main()
            self.assertEqual(error.exception.code, 2)

    def test_summarize_rejects_empty_response(self):
        with (
            patch.object(cli, "load_dotenv"),
            patch.dict(cli.os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            patch.object(cli, "load_movies", return_value=[]),
            patch.object(cli, "HybridSearch") as search,
            patch.object(cli, "OpenAI") as client,
        ):
            search.return_value.rrf_search.return_value = []
            client.return_value.chat.completions.create.return_value.choices[0].message.content = None
            with self.assertRaisesRegex(RuntimeError, "empty summary"):
                cli.summarize_command("dinosaurs")

    def test_summarize_missing_api_key(self):
        with (
            patch.object(cli, "load_dotenv"),
            patch.dict(cli.os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"),
        ):
            cli.summarize_command("dinosaurs")

    def test_citations_numbers_sources_and_limits_results(self):
        documents = [
            {"title": f"Movie {i}", "description": f"Description {i}"}
            for i in range(6)
        ]
        for limit in (5, 2):
            with self.subTest(limit=limit):
                argv = ["augmented_generation_cli.py", "citations", "action movie with lasers"]
                if limit != 5:
                    argv.extend(["--limit", str(limit)])
                output = io.StringIO()
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(cli, "load_dotenv"),
                    patch.dict(cli.os.environ, {"OPENROUTER_API_KEY": "test-key"}),
                    patch.object(cli, "load_movies", return_value=documents),
                    patch.object(cli, "HybridSearch") as search,
                    patch.object(cli, "OpenAI") as client,
                    contextlib.redirect_stdout(output),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    search.return_value.rrf_search.return_value = [
                        {"document": doc} for doc in documents
                    ]
                    completions = client.return_value.chat.completions
                    completions.create.return_value.choices[0].message.content = "  Try Movie 0 [1].  "
                    cli.main()

                search.assert_called_once_with(documents)
                search.return_value.rrf_search.assert_called_once_with(
                    "action movie with lasers", k=60, limit=limit
                )
                prompt = completions.create.call_args.kwargs["messages"][0]["content"]
                for index, doc in enumerate(documents[:limit], start=1):
                    self.assertIn(f"[{index}] Title: {doc['title']}", prompt)
                    self.assertIn(doc["description"], prompt)
                self.assertNotIn(f"Movie {limit}", prompt)
                for instruction in ("Webflyx", "[1], [2]", "different viewpoints",
                                    "I don't have enough information"):
                    self.assertIn(instruction, prompt)
                self.assertEqual(
                    output.getvalue(),
                    "Search Results:\n"
                    + "".join(f"- Movie {i}\n" for i in range(limit))
                    + "\nLLM Answer:\nTry Movie 0 [1].\n",
                )

    def test_citations_requires_query_and_positive_limit(self):
        for args in (["citations"], ["citations", "action", "--limit", "0"],
                     ["citations", "action", "--limit", "-1"]):
            with (
                self.subTest(args=args),
                patch.object(sys, "argv", ["augmented_generation_cli.py", *args]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                cli.main()
            self.assertEqual(error.exception.code, 2)
        with self.assertRaisesRegex(ValueError, "limit must be positive"):
            cli.citations_command("action", 0)

    def test_citations_rejects_empty_response(self):
        with (
            patch.object(cli, "load_dotenv"),
            patch.dict(cli.os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            patch.object(cli, "load_movies", return_value=[]),
            patch.object(cli, "HybridSearch") as search,
            patch.object(cli, "OpenAI") as client,
        ):
            search.return_value.rrf_search.return_value = []
            client.return_value.chat.completions.create.return_value.choices[0].message.content = None
            with self.assertRaisesRegex(RuntimeError, "empty citation answer"):
                cli.citations_command("action")

    def test_citations_missing_api_key(self):
        with (
            patch.object(cli, "load_dotenv"),
            patch.dict(cli.os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"),
        ):
            cli.citations_command("action")

    def test_question_retrieves_limited_context_and_prints_answer(self):
        documents = [
            {"title": f"Movie {i}", "description": f"Description {i}"}
            for i in range(6)
        ]
        question = "Who are the main characters in Jurassic Park?"
        for limit in (5, 2):
            with self.subTest(limit=limit):
                argv = ["augmented_generation_cli.py", "question", question]
                if limit != 5:
                    argv.extend(["--limit", str(limit)])
                output = io.StringIO()
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(cli, "load_dotenv"),
                    patch.dict(cli.os.environ, {"OPENROUTER_API_KEY": "test-key"}),
                    patch.object(cli, "load_movies", return_value=documents),
                    patch.object(cli, "HybridSearch") as search,
                    patch.object(cli, "OpenAI") as client,
                    contextlib.redirect_stdout(output),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    search.return_value.rrf_search.return_value = [
                        {"document": doc} for doc in documents
                    ]
                    completions = client.return_value.chat.completions
                    completions.create.return_value.choices[0].message.content = (
                        "  Alan Grant, Ellie Sattler, and Ian Malcolm.  "
                    )
                    cli.main()

                search.assert_called_once_with(documents)
                search.return_value.rrf_search.assert_called_once_with(
                    question, k=60, limit=limit
                )
                prompt = completions.create.call_args.kwargs["messages"][0]["content"]
                self.assertIn(f"Question: {question}", prompt)
                for instruction in ("Webflyx", "directly and concisely",
                                    "casual and conversational", "Don't be cringe or hype-y"):
                    self.assertIn(instruction, prompt)
                for doc in documents[:limit]:
                    self.assertIn(doc["title"], prompt)
                    self.assertIn(doc["description"], prompt)
                self.assertNotIn(f"Movie {limit}", prompt)
                self.assertEqual(
                    output.getvalue(),
                    "Search Results:\n"
                    + "".join(f"- Movie {i}\n" for i in range(limit))
                    + "\nAnswer:\nAlan Grant, Ellie Sattler, and Ian Malcolm.\n",
                )

    def test_question_requires_question_and_positive_limit(self):
        for args in (["question"], ["question", "dinosaurs", "--limit", "0"],
                     ["question", "dinosaurs", "--limit", "-1"]):
            with (
                self.subTest(args=args),
                patch.object(sys, "argv", ["augmented_generation_cli.py", *args]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                cli.main()
            self.assertEqual(error.exception.code, 2)
        with self.assertRaisesRegex(ValueError, "limit must be positive"):
            cli.question_command("dinosaurs", 0)

    def test_question_rejects_empty_response(self):
        for content in (None, "", "   "):
            with (
                self.subTest(content=content),
                patch.object(cli, "load_dotenv"),
                patch.dict(cli.os.environ, {"OPENROUTER_API_KEY": "test-key"}),
                patch.object(cli, "load_movies", return_value=[]),
                patch.object(cli, "HybridSearch") as search,
                patch.object(cli, "OpenAI") as client,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                search.return_value.rrf_search.return_value = []
                client.return_value.chat.completions.create.return_value.choices[0].message.content = content
                with self.assertRaisesRegex(RuntimeError, "empty question answer"):
                    cli.question_command("dinosaurs")

    def test_question_missing_api_key(self):
        with (
            patch.object(cli, "load_dotenv"),
            patch.dict(cli.os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"),
        ):
            cli.question_command("dinosaurs")

    def test_missing_api_key(self):
        with (
            patch.object(cli, "load_dotenv"),
            patch.dict(cli.os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"),
        ):
            cli.rag_command("dinosaurs")


if __name__ == "__main__":
    unittest.main()
