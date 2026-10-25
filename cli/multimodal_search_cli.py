import argparse

from lib.multimodal_search import image_search_command, verify_image_embedding
from lib.search_utils import DOCUMENT_PREVIEW_LENGTH


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Generate an image embedding and print its dimensions"
    )
    verify_parser.add_argument("image_path", help="Path to an image file")
    image_parser = subparsers.add_parser(
        "image_search", help="Find movies using an image"
    )
    image_parser.add_argument("image_path", help="Path to an image file")
    args = parser.parse_args()

    if args.command == "verify_image_embedding":
        verify_image_embedding(args.image_path)
    elif args.command == "image_search":
        results = image_search_command(args.image_path)
        for rank, result in enumerate(results, start=1):
            print(f"{rank}. {result['title']} (similarity: {result['similarity']:.3f})")
            print(f"   {result['description'][:DOCUMENT_PREVIEW_LENGTH]}...")
            print()


if __name__ == "__main__":
    main()
