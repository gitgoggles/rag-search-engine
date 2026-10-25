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

    def test_missing_api_key(self):
        with (
            patch.object(cli, "load_dotenv"),
            patch.dict(cli.os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"),
        ):
            cli.rag_command("dinosaurs")


if __name__ == "__main__":
    unittest.main()
