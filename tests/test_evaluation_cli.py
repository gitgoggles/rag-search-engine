import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))

import evaluation_cli


class EvaluationCliTests(unittest.TestCase):
    def test_precision_and_top_k(self):
        for limit, expected in [(3, "0.3333"), (6, "0.1667"), (5, "0.2000")]:
            with self.subTest(limit=limit):
                argv = ["evaluation_cli.py"]
                if limit != 5:
                    argv += ["--limit", str(limit)]
                dataset = {"test_cases": [{
                    "query": "cute british bear marmalade",
                    "relevant_docs": ["Paddington", "Outside top k"],
                }]}
                titles = ["Paddington"] + [f"Other {i}" for i in range(limit - 1)]
                results = [
                    {"document": {"title": title}}
                    for title in titles + ["Outside top k"]
                ]
                output = io.StringIO()
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(evaluation_cli, "load_golden_dataset", return_value=dataset),
                    patch.object(evaluation_cli, "load_movies", return_value=[]),
                    patch.object(evaluation_cli, "HybridSearch") as search,
                    contextlib.redirect_stdout(output),
                ):
                    search.return_value.rrf_search.return_value = results
                    evaluation_cli.main()
                search.return_value.rrf_search.assert_called_once_with(
                    dataset["test_cases"][0]["query"], k=60, limit=limit
                )
                self.assertIn(f"k={limit}\n", output.getvalue())
                self.assertIn(f"Precision@{limit}: {expected}", output.getvalue())
                self.assertIn(f"Retrieved: {', '.join(titles)}\n", output.getvalue())
                self.assertIn("Relevant: Paddington, Outside top k", output.getvalue())

    def test_rejects_nonpositive_limits(self):
        for limit in ["0", "-1"]:
            with (
                self.subTest(limit=limit),
                patch.object(sys, "argv", ["evaluation_cli.py", "--limit", limit]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                evaluation_cli.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
