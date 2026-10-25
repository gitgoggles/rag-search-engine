import argparse

from lib.hybrid_search import HybridSearch
from lib.search_utils import RRF_K, load_golden_dataset, load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be greater than zero")
    limit = args.limit

    dataset = load_golden_dataset()
    search = HybridSearch(load_movies())
    print(f"k={limit}")
    for test_case in dataset["test_cases"]:
        query = test_case["query"]
        results = search.rrf_search(query, k=RRF_K, limit=limit)[:limit]
        retrieved = [result["document"]["title"] for result in results]
        relevant = test_case["relevant_docs"]
        relevant_titles = set(relevant)
        precision = sum(title in relevant_titles for title in retrieved) / limit
        recall = (
            len(set(retrieved) & relevant_titles) / len(relevant_titles)
            if relevant_titles else 0.0
        )

        print(f"\n- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved)}")
        print(f"  - Relevant: {', '.join(relevant)}")


if __name__ == "__main__":
    main()
