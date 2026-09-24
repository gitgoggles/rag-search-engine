import cli.lib.keyword_search as ks
import argparse
import cli.consts as c


def main() -> None:

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")


    subparsers.add_parser("build", help="build the index")

    tf_parser = subparsers.add_parser("tf", help="find the term frequency")
    tf_parser.add_argument("doc_id", type=int, help="document id")
    tf_parser.add_argument("term", type=str, help="the term you are looking for")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    idf_parser = subparsers.add_parser("idf", help="find the idf for a term")
    idf_parser.add_argument("term", type=str, help="search term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="find the bm25idf for a term")
    bm25_idf_parser.add_argument("term", type=str, help="search term")

    tfidf_parser = subparsers.add_parser("tfidf", help="find the tf-idf")
    tfidf_parser.add_argument("doc_id", type=int, help="document id")
    tfidf_parser.add_argument("term", type=str, help="the term you are looking for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=c.BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=c.BM25_B, help="Tunable BM25 B parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs="?", default=5, help="Maximum number of results")

    args = parser.parse_args()

    match args.command:
        case "build":
            ks.build_command()
        case "tf":
            ks.tf_command(args.doc_id, args.term)
        case "search":
            ks.search_command(args)
        case "idf":
            ks.idf_command(args.term)
        case "bm25idf":
            ks.bm25_idf_command(args.term)
        case "bm25tf":
            ks.bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
        case "tfidf":
            ks.tfidf_command(args.doc_id, args.term)
        case "bm25search":
            ks.bm25_search_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
