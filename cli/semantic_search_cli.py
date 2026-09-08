from lib.semantic_search import search_command
from lib.semantic_search import embed_query_text
from lib.semantic_search import embed_text
from lib.semantic_search import verify_model
from lib.semantic_search import verify_embeddings
from lib.semantic_search import chunk_text
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="verify the model")
    subparsers.add_parser("verify_embeddings", help="verify the embeddings")

    embed_query = subparsers.add_parser("embed_query", help="embed the query")
    embed_query.add_argument("query", type=str, help="query to embed")

    embed_parser = subparsers.add_parser("embed_text", help="embed text")
    embed_parser.add_argument("text", type=str, help="text to embed")

    search_parser = subparsers.add_parser("search", help="do a search")
    search_parser.add_argument("query", type=str, help="the search query")
    search_parser.add_argument("--limit", type=int, default=5, help="result limit")

    chunk_parser = subparsers.add_parser("chunk", help="chunk some text")
    chunk_parser.add_argument("text", type=str, help="the text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="the chunk size")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.query)
        case "chunk":
            chunk_text(args.text, args.chunk_size)
        case "search":
            search_command(args.query, args.limit)
        case _:
            parser.print_help()




if __name__ == "__main__":
    main()
