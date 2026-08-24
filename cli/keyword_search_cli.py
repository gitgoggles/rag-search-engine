import math
from collections import Counter
import pickle
import os
from nltk.stem import PorterStemmer
from collections import defaultdict
import string
import json
import argparse

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
CACHE_INDEX = os.path.join(CACHE_PATH, "index.pkl")
CACHE_DOCMAP = os.path.join(CACHE_PATH, "docmap.pkl")
CACHE_TERM_FREQUENCIES = os.path.join(CACHE_PATH, "term_frequencies.pkl")
CACHE_DOC_LENGTHS = os.path.join(CACHE_PATH, "doc_lengths.pkl")

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = defaultdict(set)
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = defaultdict(int)

    def __add_document(self, doc_id, text):
        token_list = tokenize_text(text)
        
        self.doc_lengths[doc_id] = len(token_list)

        for token in token_list:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_tf(self, doc_id, term):
        if term not in self.term_frequencies[doc_id]:
            return 0
        return self.term_frequencies[doc_id][term]

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        saturated_normalized_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return saturated_normalized_tf

    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)

        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        tokenized_query = tokenize_text(query)
        scores = defaultdict(float)

        for doc_id in self.docmap:
            total = 0.0
            for token in tokenized_query:
                total += self.bm25(doc_id, token)
            scores[doc_id] = total
        sorted_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_desc[:limit]

    def get_documents(self, term):
        doc_id_set = self.index[term]
        return sorted(doc_id_set)

    def build(self):
        print("building")
        movie_list = load_movies()
        for movie in movie_list:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self):
        print("saving")
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(CACHE_INDEX, "wb") as index_file:
            pickle.dump(self.index, index_file)
        with open(CACHE_DOCMAP, "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)
        with open(CACHE_TERM_FREQUENCIES, "wb") as term_frequencies_file:
            pickle.dump(self.term_frequencies, term_frequencies_file)
        with open(CACHE_DOC_LENGTHS, "wb") as doc_lengths_file:
            pickle.dump(self.doc_lengths, doc_lengths_file)

    def load(self):
        try:
            with open(CACHE_INDEX, "rb") as index_file:
                self.index = pickle.load(index_file)
            with open(CACHE_DOCMAP, "rb") as docmap_file:
                self.docmap = pickle.load(docmap_file)
            with open(CACHE_TERM_FREQUENCIES, "rb") as term_frequencies_file:
                self.term_frequencies = pickle.load(term_frequencies_file)
            with open(CACHE_DOC_LENGTHS, "rb") as doc_lengths_file:
                self.doc_lengths = pickle.load(doc_lengths_file)
        except FileNotFoundError as e:
            print("Failed to open index and docmap caches:", e)

    def get_bm25_idf(self, term: str) -> float:
        tokenized_term = tokenize_single_term(term)

        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(tokenized_term))

        bm25_idf = math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)

        return bm25_idf

def tokenize_text(text):
    stemmer = PorterStemmer()
    translation_table = str.maketrans("", "", string.punctuation)

    # obtain stop words
    with open(STOPWORDS_PATH, "r") as stop_words_txt:
        stop_words = stop_words_txt.read().splitlines()

    lower_stop_words = list(map(lambda x: x.lower(),stop_words ))
    zero_punc_stop_words = list(map(lambda x: x.translate(translation_table), lower_stop_words ))

    # process text
    lowered = text.lower()
    punc_removed = lowered.translate(translation_table)
    word_list = punc_removed.split()
    stop_words_removed = list(filter(lambda x: x not in zero_punc_stop_words, word_list))
    stemmed = list(map(lambda x: stemmer.stem(x), stop_words_removed))

    return stemmed

def tokenize_single_term(term):
    tokenized = tokenize_text(term)
    if len(tokenized) != 1:
        raise Exception("did not return single token from tokenizer!")
    return tokenized[0]

def load_movies():
    with open(MOVIES_PATH, "r") as movies_json:
        movie_dict = json.load(movies_json)
        return movie_dict["movies"]

def build_command():
    index = InvertedIndex()
    index.build()
    index.save()


def search_command(args):
    try: 
        index = InvertedIndex()
        index.load()
    except LookupError:
        exit

    # print the search query here
    print(f"Searching for: {args.query}")
    stemmed_query = tokenize_text(args.query)

    # movie_list = load_movies()
    result_set = set()
    for term in stemmed_query:
        for doc_id in index.get_documents(term):
            result_set.add(doc_id)
            if len(result_set) >= 5:
                break
        if len(result_set) >= 5:
            break

    for result in sorted(result_set):
        movie = index.docmap[result]
        print(f"{movie["id"]}. {movie["title"]}")

def tf_command(doc_id, term):
    try: 
        index = InvertedIndex()
        index.load()
    except LookupError:
        exit
    tokenized_term = tokenize_single_term(term)
    tf = index.get_tf(doc_id, tokenized_term)
    print(f"Term frequency of '{term}': {tf}")
    return tf

def idf_command(term):
    try: 
        index = InvertedIndex()
        index.load()
    except LookupError:
        exit
    tokenized_term = tokenize_single_term(term)

    total_doc_count = len(index.docmap)
    term_match_doc_count = len(index.get_documents(tokenized_term))

    idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    print(f"Inverse document frequency of '{term}': {idf:.2f}")
    return idf

def tfidf_command(doc_id, term):
    try: 
        index = InvertedIndex()
        index.load()
    except LookupError:
        exit
    tf = tf_command(doc_id, term)
    idf = idf_command(term)

    tf_idf = tf * idf
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")

def bm25_idf_command(term):
    try: 
        index = InvertedIndex()
        index.load()
    except LookupError:
        exit

    bm25_idf = index.get_bm25_idf(term)

    print(f"BM25 Inverse document frequency of '{term}': {bm25_idf:.2f}")
    return bm25_idf

def bm25_tf_command(doc_id, term, k1=BM25_K1,b=BM25_B):
    try: 
        index = InvertedIndex()
        index.load()
    except LookupError:
        exit
    tokenized_term = tokenize_single_term(term)
    bm25tf = index.get_bm25_tf(doc_id, tokenized_term, k1, b)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")
    return bm25tf

def bm25_search_command(query, limit=5):
    try: 
        index = InvertedIndex()
        index.load()
    except LookupError:
        exit
    result_tuples = index.bm25_search(query, limit)

    num = 1
    for doc_id, score in result_tuples:
        movie = index.docmap[doc_id]

        print(f"{num}. ({movie['id']}) {movie['title']} - Score: {score:.2f}")
        num += 1




def main() -> None:

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")


    subparsers.add_parser("build", help="build the index")

    tf_parser = subparsers.add_parser("tf", help="find the term frequency")
    tf_parser.add_argument("doc_id", type=int, help="document id")
    tf_parser.add_argument("term", type=str, help="the term you are looking for")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    idf_parser = subparsers.add_parser("idf", help="find the idf for a term")
    idf_parser.add_argument("term", type=str, help="search term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="find the bm25idf for a term")
    bm25_idf_parser.add_argument("term", type=str, help="search term")

    tfidf_parser = subparsers.add_parser("tfidf", help="find the tf-idf")
    tfidf_parser.add_argument("doc_id", type=int, help="document id")
    tfidf_parser.add_argument("term", type=str, help="the term you are looking for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 B parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs="?", default=5, help="Maximum number of results")

    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "search":
            search_command(args)
        case "idf":
            idf_command(args.term)
        case "bm25idf":
            bm25_idf_command(args.term)
        case "bm25tf":
            bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case "bm25search":
            bm25_search_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
