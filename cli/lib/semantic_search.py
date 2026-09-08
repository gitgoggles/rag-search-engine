import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import os
import numpy as np
from consts import PROJECT_ROOT


CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
CACHE_EMBEDDINGS = os.path.join(CACHE_PATH, "movie_embeddings.npy")
MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
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
        doc_list = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
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


def chunk_text(text: str, chunk_size: int):
    print(f"Chunking {len(text)} characters")

    if chunk_size <= 0:                                                        
           raise ValueError("chunk_size must be positive")                        
                                                                                  
    words = text.split()                                                       
    chunk_list: list[str] = []                                                 
                                                                              
    for start in range(0, len(words), chunk_size):                             
       chunk = " ".join(words[start:start + chunk_size])                      
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
