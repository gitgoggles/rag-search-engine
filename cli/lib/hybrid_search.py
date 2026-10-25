import json
import os
import time

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


def rewrite_query(query: str) -> str:
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
                "content": f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
""",
            }
        ],
    )
    return (response.choices[0].message.content or "").strip() or query


def expand_query(query: str) -> str:
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
                "content": f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
""",
            }
        ],
    )
    return (response.choices[0].message.content or "").strip() or query


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


def rerank_individual(query: str, results: list[dict]) -> list[dict]:
    if not results:
        return []

    load_dotenv()
    client = OpenAI(
        base_url=os.environ.get("RERANK_BASE_URL", "https://llama.rucknet.uk/v1"),
        api_key=os.environ.get("RERANK_API_KEY") or "unused",
    )
    reranked = []
    for index, result in enumerate(results):
        if index:
            time.sleep(3)
        doc = result["document"]
        response = client.chat.completions.create(
            model=os.environ.get("RERANK_MODEL", "Qwen3.6-35B-A3B-UD-Q4_K_M"),
            max_tokens=32,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            messages=[
                {
                    "role": "system",
                    "content": f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:""",
                }
            ],
        )
        content = response.choices[0].message.content
        try:
            score = float(content or "")
        except ValueError as exc:
            raise ValueError(f"Invalid re-rank score: {content!r}") from exc
        if not 0 <= score <= 10:
            raise ValueError(f"Re-rank score must be between 0 and 10: {content!r}")
        reranked.append({**result, "rerank_score": score})

    return sorted(reranked, key=lambda result: result["rerank_score"], reverse=True)


def rerank_batch(query: str, results: list[dict]) -> list[dict]:
    if not results:
        return []

    load_dotenv()
    client = OpenAI(
        base_url=os.environ.get("RERANK_BASE_URL", "https://llama.rucknet.uk/v1"),
        api_key=os.environ.get("RERANK_API_KEY") or "unused",
    )
    doc_list_str = json.dumps([result["document"] for result in results])
    response = client.chat.completions.create(
        model=os.environ.get("RERANK_MODEL", "Qwen3.6-35B-A3B-UD-Q4_K_M"),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        messages=[
            {
                "role": "user",
                "content": f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return all the movie IDs in order of relevance, best match first.
Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text. Include each movie ID exactly once.

For example:
[75, 12, 34, 2, 1]

Ranking:""",
            }
        ],
    )
    content = response.choices[0].message.content
    try:
        ranked_ids = json.loads(content or "")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid batch re-rank response: {content!r}") from exc
    documents = {result["document"]["id"]: result for result in results}
    if (
        not isinstance(ranked_ids, list)
        or any(type(doc_id) is not int for doc_id in ranked_ids)
        or len(ranked_ids) != len(documents)
        or set(ranked_ids) != set(documents)
    ):
        raise ValueError("Batch re-ranking must return each candidate movie ID exactly once")

    ranks = {doc_id: rank for rank, doc_id in enumerate(ranked_ids, start=1)}
    return sorted(
        [{**result, "rerank_rank": ranks[result["document"]["id"]]} for result in results],
        key=lambda result: result["rerank_rank"],
    )


def rrf_search_command(query, k, limit, enhance=None, rerank_method=None):
    if enhance == "spell":
        enhanced_query = correct_spelling(query)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
        query = enhanced_query
    if enhance == "rewrite":
        rewritten_query = rewrite_query(query)
        print(f"Rewritten query ({enhance}): '{query}' -> '{rewritten_query}'\n")
        query = rewritten_query
    if enhance == "expand":
        expanded_query = expand_query(query)
        print(f"Rewritten query ({enhance}): '{query}' -> '{expanded_query}'\n")
        query = expanded_query

    search = HybridSearch(load_movies())
    candidate_limit = limit * 5 if rerank_method else limit
    results = search.rrf_search(query, k, candidate_limit)
    if rerank_method:
        print(f"Re-ranking top {limit} results using {rerank_method} method...")
        rerank = rerank_batch if rerank_method == "batch" else rerank_individual
        results = rerank(query, results[:max(0, candidate_limit)])
        print(f"Reciprocal Rank Fusion Results for '{query}' (k={k})\n")
    for rank, result in enumerate(results[:max(0, limit)], start=1):
        document = result["document"]
        print(f"{rank}. {document['title']}")
        if rerank_method == "individual":
            print(f"  Re-rank Score: {result['rerank_score']:.3f}/10")
        if rerank_method == "batch":
            print(f"  Re-rank Rank: {result['rerank_rank']}")
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
