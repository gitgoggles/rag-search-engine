import argparse
import lib.hybrid_search as hs

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    norm_parser = subparsers.add_parser("normalize", help="normalize a list of floats", )
    norm_parser.add_argument("values", nargs="*", type=float)

    args = parser.parse_args()

    match args.command:
        case "normalize":
            hs.normalize_command(args.values)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
