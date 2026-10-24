import os

from dotenv import load_dotenv
from openai import OpenAI

from .keyword_search import CACHE_INDEX, InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import DOCUMENT_PREVIEW_LENGTH, load_movies


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(CACHE_INDEX):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if limit <= 0:
            return []

        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_scores = normalize([score for _, score in bm25_results])
        semantic_scores = normalize([result["score"] for result in semantic_results])

        documents = {doc["id"]: doc for doc in self.documents}
        merged = {}

        def get_result(doc_id):
            if doc_id not in merged:
                merged[doc_id] = {
                    "document": documents[doc_id],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }
            return merged[doc_id]

        for (doc_id, _), score in zip(bm25_results, bm25_scores):
            get_result(doc_id)["bm25_score"] = score

        for result, score in zip(semantic_results, semantic_scores):
            get_result(result["id"])["semantic_score"] = score

        for result in merged.values():
            result["hybrid_score"] = hybrid_score(
                result["bm25_score"], result["semantic_score"], alpha
            )

        return sorted(
            merged.values(), key=lambda result: result["hybrid_score"], reverse=True
        )

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        if limit <= 0:
            return []

        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        documents = {doc["id"]: doc for doc in self.documents}
        merged = {}

        for rank_key, doc_ids in (
            ("bm25_rank", [doc_id for doc_id, _ in bm25_results]),
            ("semantic_rank", [result["id"] for result in semantic_results]),
        ):
            for rank, doc_id in enumerate(doc_ids, start=1):
                if doc_id not in merged:
                    merged[doc_id] = {
                        "document": documents[doc_id],
                        "bm25_rank": None,
                        "semantic_rank": None,
                        "rrf_score": 0.0,
                    }
                result = merged[doc_id]
                result[rank_key] = rank
                result["rrf_score"] += rrf_score(rank, k)

        return sorted(
            merged.values(), key=lambda result: result["rrf_score"], reverse=True
        )


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)

    return [(score - min_score) / (max_score - min_score) for score in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize_command(float_list):
    for score in normalize(float_list):
        print(f"* {score:.4f}")


def correct_spelling(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query.
Output only the final query text, nothing else.
User query: "{query}"
""",
            }
        ],
    )
    return (response.choices[0].message.content or "").strip() or query


def rrf_search_command(query, k, limit, enhance=None):
    if enhance == "spell":
        enhanced_query = correct_spelling(query)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
        query = enhanced_query

    search = HybridSearch(load_movies())
    results = search.rrf_search(query, k, limit)
    for rank, result in enumerate(results[:limit], start=1):
        document = result["document"]
        print(f"{rank}. {document['title']}")
        print(f"  RRF Score: {result['rrf_score']:.3f}")
        print(
            f"  BM25 Rank: {result['bm25_rank']}, "
            f"Semantic Rank: {result['semantic_rank']}"
        )
        print(f"  {document['description'][:DOCUMENT_PREVIEW_LENGTH]}")
        print()


def weighted_search_command(query, alpha, limit):
    search = HybridSearch(load_movies())
    results = search.weighted_search(query, alpha, limit)
    for rank, result in enumerate(results[:limit], start=1):
        document = result["document"]
        print(f"{rank}. {document['title']}")
        print(f"Hybrid Score: {result['hybrid_score']:.3f}")
        print(
            f"BM25: {result['bm25_score']:.3f}, "
            f"Semantic: {result['semantic_score']:.3f}"
        )
        print(document["description"][:DOCUMENT_PREVIEW_LENGTH])
