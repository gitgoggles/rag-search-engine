import argparse
import lib.hybrid_search as hs

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # normalize
    norm_parser = subparsers.add_parser("normalize", help="normalize a list of floats", )
    norm_parser.add_argument("values", nargs="*", type=float)

    # weighted search
    weighted_parser = subparsers.add_parser("weighted-search", help="perform a weighted search", )
    weighted_parser.add_argument("query", type=str)
    weighted_parser.add_argument("--alpha", default=0.5, type=float)
    weighted_parser.add_argument("--limit", default=5, type=int)

    args = parser.parse_args()

    match args.command:
        case "normalize":
            hs.normalize_command(args.values)
        case "weighted-search":
            hs.weighted_search_command(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
