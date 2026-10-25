from PIL import Image
from sentence_transformers import SentenceTransformer

from .search_utils import DEFAULT_SEARCH_LIMIT, Movie, load_movies
from .semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(
        self, documents: list[Movie] | None = None, model_name: str = "clip-ViT-B-32"
    ):
        self.model = SentenceTransformer(model_name)
        self.documents = documents if documents is not None else []
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = (
            self.model.encode(self.texts, show_progress_bar=True) if self.texts else []
        )

    def embed_image(self, image_path: str):
        with Image.open(image_path) as image:
            return self.model.encode([image])[0]

    def search_with_image(self, image_path: str) -> list[dict]:
        image_embedding = self.embed_image(image_path)
        results = [
            {
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "similarity": float(cosine_similarity(text_embedding, image_embedding)),
            }
            for doc, text_embedding in zip(self.documents, self.text_embeddings, strict=True)
        ]
        return sorted(results, key=lambda result: result["similarity"], reverse=True)[
            :DEFAULT_SEARCH_LIMIT
        ]


def image_search_command(image_path: str) -> list[dict]:
    search = MultimodalSearch(load_movies())
    return search.search_with_image(image_path)


def verify_image_embedding(image_path: str) -> None:
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
