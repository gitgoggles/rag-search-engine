import cli.lib.semantic_search as ss
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="verify the model")
    subparsers.add_parser("verify_embeddings", help="verify the embeddings")
    subparsers.add_parser("embed_chunks", help="embed the query")

    embed_query = subparsers.add_parser("embed_query", help="embed the query")
    embed_query.add_argument("query", type=str, help="query to embed")

    embed_parser = subparsers.add_parser("embed_text", help="embed text")
    embed_parser.add_argument("text", type=str, help="text to embed")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="do a chunked search")
    search_chunked_parser.add_argument("query", type=str, help="the search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="result limit")

    search_parser = subparsers.add_parser("search", help="do a search")
    search_parser.add_argument("query", type=str, help="the search query")
    search_parser.add_argument("--limit", type=int, default=5, help="result limit")

    chunk_parser = subparsers.add_parser("chunk", help="chunk some text")
    chunk_parser.add_argument("text", type=str, help="the text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="the chunk size")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="define overlap size")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="semanticly chunk some text")
    semantic_chunk_parser.add_argument("text", type=str, help="the text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="the chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="define overlap size")

    args = parser.parse_args()

    match args.command:
        case "embed_chunks":
            ss.embed_chunks()
        case "verify":
            ss.verify_model()
        case "verify_embeddings":
            ss.verify_embeddings()
        case "embed_text":
            ss.embed_text(args.text)
        case "embed_query":
            ss.embed_query_text(args.query)
        case "chunk":
            ss.chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            ss.semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "search":
            ss.search_command(args.query, args.limit)
        case "search_chunked":
            ss.search_chunked_command(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
