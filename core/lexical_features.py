import os
import nltk
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
from taaled import ld
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from core.sentence_parser import SentenceParser
from core.utils import get_user_by_platform
import tensorflow_hub as hub

nltk.download("wordnet")


def is_noun(token: str) -> bool:
    nouns = {x.name().split(".", 1)[0] for x in wn.all_synsets("n")}
    return token in nouns


def is_verb(token: str) -> bool:
    classification = wn.synsets(token)
    print(classification)
    # Sometimes the classidication will be an empty list so we have to explicitly check for that case and just return False
    if len(classification) > 0:
        return classification[0].pos() == "v"
    return False


def is_adjective(token: str):
    classification = wn.synsets(token)
    # Sometimes the classidication will be an empty list so we have to explicitly check for that case and just return False
    if len(classification) > 0:
        return classification[0].pos() == "a"
    return False


def is_adverb(token: str):
    classification = wn.synsets(token)
    # Sometimes the classidication will be an empty list so we have to explicitly check for that case and just return False
    if len(classification) > 0:
        return classification[0].pos() == "r"
    return False


def is_lexical_word(token: str):
    "A lexical word is defined such as nouns, adjectives, verbs, and adverbs that convey meaning in a text"
    if is_verb(token) or is_adjective(token) or is_adverb(token) or is_noun(token):
        return True
    return False


def lexical_diversity(tokens):
    "Take the number of lexical words divided by the total word count"
    word_count = len(tokens)
    lexical_word_count = 0
    for token in tokens:
        print("token:", token)
        if is_lexical_word(token):
            lexical_word_count += 1
        return lexical_word_count / word_count


# Refrence implementation: hthttps://lcr-ads-lab.github.io/TAALED/ld_indices/1.%20Revised_LD_indices.html#mattr


# The Moving-Average Type-Token Ratio calculates the moving average for all
# segments of a given length. For a segment length of 50 tokens, TTR is
# calculated on tokens 1-50, 2-51, 3-52, etc., and the resulting TTR
# measurements are averaged to produce the final MATTR value
def mattr(user_id, platform_id, session_id=None):
    sp = SentenceParser(os.path.join(os.getcwd(), "cleaned2.csv"))
    df = get_user_by_platform(user_id, platform_id, session_id)
    key_set = list(df["key"])
    words = sp.get_words(key_set)
    ldvals = ld.lexdiv(words)
    print("MATTR:", ldvals.mattr)


# Refrence implementation: https://lcr-ads-lab.github.io/TAALED/ld_indices/1.%20Revised_LD_indices.html#mtld

# MTLD measures the average number of tokens it takes for the TTR value to reach
# a point of stabilization (which McCarthy & Jarvis, 2010 defined as TTR = .720).
# Texts that are less lexically diverse will have lower MTLD values than more
# lexically diverse texts.

# MTLD is calculated by determining the number of factors in a text (the number
# of non-overlapping portions of text that reach TTR = .720) and the length of
# each factor. In most texts, a partial factor will occur at the end of the
# text. MTLD runs both forwards and backwards.


def mtldo(user_id, platform_id, session_id=None):
    sp = SentenceParser(os.path.join(os.getcwd(), "cleaned2.csv"))
    df = get_user_by_platform(user_id, platform_id, session_id)
    key_set = list(df["key"])
    words = sp.get_words(key_set)
    print("Words:", words)
    ldvals = ld.lexdiv(words)
    print("MATTR:", ldvals.mattr)


def bert_similarity(text1, text2):
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)
    return util.pytorch_cos_sim(embedding1, embedding2)


# Not very good it seems like to high similarity scores for different users
def word_movers_distance(text1_arr, text2_arr):
    # Train Word2Vec model
    model = Word2Vec(
        [text1_arr, text2_arr], vector_size=100, window=5, min_count=1, workers=4
    )

    # Use the model's KeyedVectors instance to calculate WMD
    similarity = model.wv.wmdistance(text1_arr, text2_arr)

    return similarity


def USE_similarity(text1, text2):
    texts = [text1, text2]
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embeddings = model(texts)
    return cosine_similarity(np.array(embeddings))


def lsa_similarity(text1, text2):
    texts = [text1, text2]
    vectorizer = TfidfVectorizer()

    # Fit and transform the texts to TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Apply LSA
    svd = TruncatedSVD(n_components=100)
    lsa_matrix = svd.fit_transform(tfidf_matrix)

    # Compute cosine similarity
    cosine_sim = cosine_similarity(lsa_matrix[0:1], lsa_matrix[1:2])
    return cosine_sim[0][0]
