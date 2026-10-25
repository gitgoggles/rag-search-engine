import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))
import multimodal_search_cli as cli
from lib import multimodal_search as ms


class MultimodalSearchTests(unittest.TestCase):
    def test_model_name(self):
        with patch.object(ms, "SentenceTransformer") as model:
            search = ms.MultimodalSearch()
            model.assert_called_once_with("clip-ViT-B-32")
            self.assertIs(search.model, model.return_value)
            ms.MultimodalSearch(model_name="custom-model")
            model.assert_called_with("custom-model")

    def test_document_embeddings(self):
        documents = [{"id": 42, "title": "Paddington", "description": "A bear in London"}]
        with patch.object(ms, "SentenceTransformer") as model:
            search = ms.MultimodalSearch(documents)
            self.assertIs(search.documents, documents)
            self.assertEqual(search.texts, ["Paddington: A bear in London"])
            model.return_value.encode.assert_called_once_with(
                search.texts, show_progress_bar=True
            )
            self.assertIs(search.text_embeddings, model.return_value.encode.return_value)

    def test_search_with_image_ranks_by_cosine_and_limits_to_five(self):
        documents = [
            {"id": i, "title": f"Movie {i}", "description": f"Description {i}"}
            for i in range(7)
        ]
        embeddings = np.array([[1, i] for i in range(7)])
        with patch.object(ms, "SentenceTransformer") as model:
            model.return_value.encode.return_value = embeddings
            search = ms.MultimodalSearch(documents)
            with patch.object(search, "embed_image", return_value=np.array([0, 2])) as embed:
                results = search.search_with_image("poster.jpeg")
                embed.assert_called_once_with("poster.jpeg")
        self.assertEqual([result["id"] for result in results], [6, 5, 4, 3, 2])
        self.assertEqual(results[0]["title"], "Movie 6")
        self.assertEqual(results[0]["description"], "Description 6")
        self.assertAlmostEqual(results[0]["similarity"], 6 / np.sqrt(37))

    def test_empty_documents(self):
        with patch.object(ms, "SentenceTransformer") as model:
            search = ms.MultimodalSearch([])
            model.return_value.encode.assert_not_called()
            with patch.object(search, "embed_image", return_value=np.zeros(2)):
                self.assertEqual(search.search_with_image("poster.jpeg"), [])

    def test_zero_embedding(self):
        documents = [{"id": 1, "title": "Movie", "description": "Description"}]
        with patch.object(ms, "SentenceTransformer") as model:
            model.return_value.encode.return_value = [np.zeros(2)]
            search = ms.MultimodalSearch(documents)
            with patch.object(search, "embed_image", return_value=np.ones(2)):
                self.assertEqual(search.search_with_image("poster.jpeg")[0]["similarity"], 0)

    def test_image_search_command(self):
        with patch.object(ms, "load_movies") as movies, patch.object(ms, "MultimodalSearch") as search:
            results = ms.image_search_command("poster.jpeg")
            search.assert_called_once_with(movies.return_value)
            search.return_value.search_with_image.assert_called_once_with("poster.jpeg")
            self.assertIs(results, search.return_value.search_with_image.return_value)

    def test_image_search_cli(self):
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["multimodal_search_cli.py", "image_search", "poster.jpeg"]),
            patch.object(cli, "image_search_command") as command,
            contextlib.redirect_stdout(output),
        ):
            command.return_value = [
                {"id": 42, "title": "Paddington", "description": "A" * 120, "similarity": 0.3094}
            ]
            cli.main()
            command.assert_called_once_with("poster.jpeg")
        self.assertEqual(output.getvalue(), f"1. Paddington (similarity: 0.309)\n   {'A' * 100}...\n\n")

    def test_embed_image(self):
        with (
            patch.object(ms, "SentenceTransformer") as model,
            patch.object(ms.Image, "open") as open_image,
        ):
            image = open_image.return_value.__enter__.return_value
            embedding = np.zeros(512)
            model.return_value.encode.return_value = [embedding]
            search = ms.MultimodalSearch()
            self.assertIs(search.embed_image("poster.jpeg"), embedding)
            open_image.assert_called_once_with("poster.jpeg")
            model.return_value.encode.assert_called_once_with([image])
            open_image.return_value.__exit__.assert_called_once()

    def test_verify_image_embedding(self):
        output = io.StringIO()
        with patch.object(ms, "MultimodalSearch") as search, contextlib.redirect_stdout(output):
            search.return_value.embed_image.return_value = np.zeros(512)
            ms.verify_image_embedding("poster.jpeg")
            search.return_value.embed_image.assert_called_once_with("poster.jpeg")
        self.assertEqual(output.getvalue(), "Embedding shape: 512 dimensions\n")

    def test_cli_dispatch(self):
        with (
            patch.object(sys, "argv", ["multimodal_search_cli.py", "verify_image_embedding", "poster.jpeg"]),
            patch.object(cli, "verify_image_embedding") as verify,
        ):
            cli.main()
            verify.assert_called_once_with("poster.jpeg")

    def test_required_arguments(self):
        for args in [[], ["verify_image_embedding"], ["image_search"]]:
            with (
                self.subTest(args=args),
                patch.object(sys, "argv", ["multimodal_search_cli.py", *args]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                cli.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
