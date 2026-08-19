import math
from collections import Counter
import pickle
import os
from nltk.stem import PorterStemmer
from collections import defaultdict
import string
import json
import argparse

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = defaultdict(set)
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id, text):
        token_list = tokenize_text(text)
        for token in token_list:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_tf(self, doc_id, term):
        if term not in self.term_frequencies[doc_id]:
            return 0
        return self.term_frequencies[doc_id][term]

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

    def load(self):
        try:
            with open(CACHE_INDEX, "rb") as index_file:
                self.index = pickle.load(index_file)
            with open(CACHE_DOCMAP, "rb") as docmap_file:
                self.docmap = pickle.load(docmap_file)
            with open(CACHE_TERM_FREQUENCIES, "rb") as term_frequencies_file:
                self.term_frequencies = pickle.load(term_frequencies_file)
        except FileNotFoundError as e:
            print("Failed to open index and docmap caches:", e)




PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
CACHE_INDEX = os.path.join(CACHE_PATH, "index.pkl")
CACHE_DOCMAP = os.path.join(CACHE_PATH, "docmap.pkl")
CACHE_TERM_FREQUENCIES = os.path.join(CACHE_PATH, "term_frequencies.pkl")

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

    tfidf_parser = subparsers.add_parser("tfidf", help="find the tf-idf")
    tfidf_parser.add_argument("doc_id", type=int, help="document id")
    tfidf_parser.add_argument("term", type=str, help="the term you are looking for")

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
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
