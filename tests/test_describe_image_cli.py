import base64
import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))
import describe_image_cli as cli


class DescribeImageTests(unittest.TestCase):
    def run_cli(self, image="poster.jpeg", content=" Paddington in London ", usage=True):
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["describe_image_cli.py", "--image", image,
                                      "--query", " bear in London "]),
            patch.object(cli, "load_dotenv"),
            patch.dict(cli.os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            patch.object(cli, "open", mock_open(read_data=b"image bytes"), create=True) as image_file,
            patch.object(cli, "OpenAI") as client,
            contextlib.redirect_stdout(output),
        ):
            client.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(total_tokens=123) if usage else None,
            )
            cli.main()
        image_file.assert_called_once_with(image, "rb")
        client.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1", api_key="test-key"
        )
        return output.getvalue(), client.return_value.chat.completions.create.call_args.kwargs

    def test_multimodal_request_and_output(self):
        output, request = self.run_cli()
        self.assertEqual(output, "Rewritten query: Paddington in London\nTotal tokens: 123\n")
        self.assertEqual(request["model"], "openrouter/free")
        message, = request["messages"]
        self.assertEqual(message["role"], "user")
        self.assertEqual(message["content"][1], {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b"image bytes").decode()},
        })
        self.assertEqual(message["content"][2], {"type": "text", "text": "bear in London"})

    def test_mime_detection_and_fallback(self):
        for filename, mime in [("poster.png", "image/png"), ("poster", "image/jpeg")]:
            with self.subTest(filename=filename):
                _, request = self.run_cli(image=filename)
                url = request["messages"][0]["content"][1]["image_url"]["url"]
                self.assertTrue(url.startswith(f"data:{mime};base64,"))

    def test_missing_usage(self):
        output, _ = self.run_cli(usage=False)
        self.assertNotIn("Total tokens:", output)

    def test_missing_content(self):
        with self.assertRaisesRegex(RuntimeError, "No text in API response"):
            self.run_cli(content=None)

    def test_required_arguments(self):
        for args in [[], ["--image", "poster.jpeg"], ["--query", "bear"]]:
            with (
                self.subTest(args=args),
                patch.object(sys, "argv", ["describe_image_cli.py", *args]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                cli.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
