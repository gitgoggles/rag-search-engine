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
            ms.MultimodalSearch("custom-model")
            model.assert_called_with("custom-model")

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
        for args in [[], ["verify_image_embedding"]]:
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
