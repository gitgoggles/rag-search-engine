import argparse

from lib.multimodal_search import verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Generate an image embedding and print its dimensions"
    )
    verify_parser.add_argument("image_path", help="Path to an image file")
    args = parser.parse_args()

    if args.command == "verify_image_embedding":
        verify_image_embedding(args.image_path)


if __name__ == "__main__":
    main()
