import os

from .keyword_search import CACHE_INDEX, InvertedIndex
from .semantic_search import ChunkedSemanticSearch


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
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_command(arg_list):
    if not arg_list:
        return

    min_score = min(arg_list)
    max_score = max(arg_list)

    if min_score == max_score:
        for num in arg_list:
            print(f"* {1.0:.4f}")
        return

    for score in arg_list:
        normalized = (score - min_score) / (max_score - min_score)
        print(f"* {normalized:.4f}")


