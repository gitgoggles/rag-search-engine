import argparse
import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite a movie query using an image")
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--query", required=True, help="Text query to rewrite")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    with open(args.image, "rb") as image_file:
        img = image_file.read()
    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"

    system_prompt = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }
    ]
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    response = client.chat.completions.create(
        model="openrouter/free", messages=messages
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("No text in API response")

    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
