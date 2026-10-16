import os

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
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

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
