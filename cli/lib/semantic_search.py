from .search_utils import format_search_result
import re
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import cli.consts as c

CACHE_PATH = os.path.join(c.PROJECT_ROOT, "cache")
CACHE_EMBEDDINGS = os.path.join(CACHE_PATH, "movie_embeddings.npy")
CACHE_CHUNK_EMBEDDINGS = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
CACHE_CHUNK_METADATA = os.path.join(CACHE_PATH, "chunk_metadata.json")
MOVIES_PATH = os.path.join(c.PROJECT_ROOT, "data", "movies.json")

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = defaultdict()

    def build_embeddings(self,documents):
        self.documents = documents
        doc_list = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        np.save(CACHE_EMBEDDINGS, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(CACHE_EMBEDDINGS):
            self.embeddings = np.load(CACHE_EMBEDDINGS)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        else:
            return self.build_embeddings(documents)


    def generate_embedding(self, text):
        if not text.strip() or not text:
            raise ValueError('input text is empty')

        encoded = self.model.encode([text.strip()])
        return encoded[0]

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("embeddings aren't loaded!")
        query_emb = self.generate_embedding(query)
        score_list = []
        for doc, doc_emb in zip(self.documents, self.embeddings):
            similarity = cosine_similarity(query_emb, doc_emb)
            score_list.append((similarity, doc))
        score_list_desc = sorted(score_list, key=lambda x: x[0], reverse=True)
        list_of_dicts = []
        for result in score_list_desc[:limit]:
            list_of_dicts.append({"score": result[0], "title": result[1]["title"], "description": result[1]["description"]})
        return list_of_dicts

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        chunk_list: list[str] = []
        chunk_metadata_list: list[dict] = []

        for doc_i, doc in enumerate(documents):
            if doc['description'] == "":
                continue
            inner_chunk_list = semantic_chunk_text(doc['description'], 4, 1)
            chunk_list.extend(inner_chunk_list)
            for chunk_i, _ in enumerate(inner_chunk_list):
                chunk_metadata_list.append({"movie_idx": doc_i, "chunk_idx": chunk_i, "total_chunks": len(inner_chunk_list)})

        self.chunk_embeddings = self.model.encode(chunk_list, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata_list

        np.save(CACHE_CHUNK_EMBEDDINGS, self.chunk_embeddings)
        with open(CACHE_CHUNK_METADATA, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunks": self.chunk_metadata,
                    "total_chunks": len(self.chunk_embeddings)
                },
                f,
                indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CACHE_CHUNK_EMBEDDINGS) and os.path.exists(CACHE_CHUNK_METADATA):
            self.chunk_embeddings = np.load(CACHE_CHUNK_EMBEDDINGS)

            with open(CACHE_CHUNK_METADATA, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]

            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

# you are here <-------------
    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunk_score_list: list[dict] = []
        best_chunk_scores = {}

        for idx, chunk_embedding in enumerate(self.chunk_embeddings):
            movie_idx = self.chunk_metadata[idx]["movie_idx"]
            score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_dict = {"chunk_idx": idx, "movie_idx": movie_idx, "score": score}
            chunk_score_list.append(chunk_dict)

            if movie_idx not in best_chunk_scores:
                best_chunk_scores[movie_idx] = chunk_dict

            if best_chunk_scores[movie_idx]["score"] < score:
                best_chunk_scores[movie_idx] = chunk_dict

        sorted_best_chunk_scores = list(sorted(best_chunk_scores.items(), key=lambda x: x[1]["score"], reverse=True))
        limited_best_chunk_scores = sorted_best_chunk_scores[:limit]

        result_list = []
        for chunk in limited_best_chunk_scores:
            doc = self.documents[chunk[1]["movie_idx"]]
            formatted = format_search_result(doc["id"], doc["title"], doc["description"][:100], chunk[1]["score"])
            result_list.append(formatted)

        return result_list


def embed_chunks():
    with open(MOVIES_PATH, "r") as movies_json:
        movie_list: list = json.load(movies_json)["movies"]
        chunked_sem_search = ChunkedSemanticSearch()
        chunked_sem_search.load_or_create_chunk_embeddings(movie_list)
        print(f"Generated {len(chunked_sem_search.chunk_embeddings)} chunked embeddings")




def semantic_chunk_text(text: str, max_chunk_size: int, overlap: int):
    print(f"Semantically chunking {len(text)} characters")


    if max_chunk_size <= 0:                                                        
           raise ValueError("max_chunk_size must be positive")                        

    if max_chunk_size - overlap <= 0:                                                        
           raise ValueError("(max_chunk_size - overlap) must be positive")                        

    text = text.strip()

    # If there's nothing left after stripping, return an empty list.
    if not text:
        return []

    sentence_list = re.split(r"(?<=[.!?])\s+", text)

    chunk_list: list[str] = []                                                 
                                                                              
    step = max_chunk_size - overlap
    stop = max(1, len(sentence_list) - overlap)

    for start in range(0, stop, step):

       sentences = sentence_list[start:start + max_chunk_size]
       chunk = " ".join(sentences).strip()

       if not chunk:
           continue

       chunk_list.append(chunk)                                               
                                                                              
    for index, chunk in enumerate(chunk_list, start=1):                        
       print(f"{index}. {chunk}")                                             
                                                                              
    return chunk_list   

def chunk_text(text: str, chunk_size: int, overlap: str):
    print(f"Chunking {len(text)} characters")

    if chunk_size <= 0:                                                        
           raise ValueError("chunk_size must be positive")                        
                                                                                  
    words = text.split()                                                       
    chunk_list: list[str] = []                                                 
                                                                              
    for start in range(0, len(words), chunk_size):                             
       if start - overlap < 0:
           chunk = " ".join(words[start:start + chunk_size])                      

       if start - overlap >= 0:
           chunk = " ".join(words[start - overlap:start + chunk_size])                      

       chunk_list.append(chunk)                                               
                                                                              
    for index, chunk in enumerate(chunk_list, start=1):                        
       print(f"{index}. {chunk}")                                             
                                                                              
    return chunk_list   


def search_command(query, limit):
    semantic_search = SemanticSearch()
    with open(MOVIES_PATH, "r") as movies_json:
        movie_list: list = json.load(movies_json)["movies"]
        semantic_search.load_or_create_embeddings(movie_list)
        results_list = semantic_search.search(query, limit)
        i = 1
        for result in results_list:
            print(f"{i}. {result["title"]} (score: {result["score"]})\n{result["description"]}")
            i += 1

def search_chunked_command(query, limit):
    chunked_semantic_search = ChunkedSemanticSearch()
    with open(MOVIES_PATH, "r") as movies_json:
        movie_list: list = json.load(movies_json)["movies"]
        chunked_semantic_search.load_or_create_chunk_embeddings(movie_list)
        results_list = chunked_semantic_search.search_chunks(query, limit)
        for i, result in enumerate(results_list, start=1):
            print(f"{i}. {result['title']} (score: {result['score']})\n{result['document']}")




def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    with open(MOVIES_PATH, "r") as movies_json:
        movie_list: list = json.load(movies_json)["movies"]
        semantic_search.load_or_create_embeddings(movie_list)
        documents = semantic_search.documents
        embeddings = semantic_search.embeddings
        print(f"Number of docs:   {len(documents)}")
        print(
            f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
        )


def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    semantic_search = SemanticSearch()
    model = semantic_search.model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")
