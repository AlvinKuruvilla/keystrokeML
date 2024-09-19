import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from tqdm import tqdm

from core.lexical_features import (
    USE_similarity,
    bert_similarity,
    extract_stylometric_features,
    lsa_similarity,
    mtldo,
    word_movers_distance,
)
from core.sentence_parser import SentenceParser, reconstruct_text
from core.utils import (
    all_ids,
    get_df_by_user_id,
    get_user_by_platform,
    get_user_by_platform_from_df,
    map_platform_id_to_initial,
    read_compact_format,
)


def simple_bert():
    df = get_user_by_platform(1, 3)
    df2 = get_user_by_platform(1, 2)
    key_set = list(df["key"])
    sent1 = reconstruct_text(key_set)

    key_set2 = list(df2["key"])
    sent2 = reconstruct_text(key_set2)
    print(float(bert_similarity(sent1, sent2)))
    sentences = reconstruct_text(key_set)
    print(sentences)

    mtldo(1, 2)


def full_bert():
    raw_data = read_compact_format()
    for uid in all_ids():
        for other in all_ids():
            df = get_df_by_user_id(raw_data, uid)
            df2 = get_df_by_user_id(raw_data, other)
            key_set = list(df["key"])
            sent1 = reconstruct_text(key_set)
            key_set2 = list(df2["key"])
            sent2 = reconstruct_text(key_set2)
            print(f"User {uid} and {other}:", float(bert_similarity(sent1, sent2)))


def simple_wmd():
    sp = SentenceParser(os.path.join(os.getcwd(), "cleaned2.csv"))
    df = get_user_by_platform(1, 3)
    df2 = get_user_by_platform(4, 2)
    key_set = list(df["key"])
    key_set2 = list(df2["key"])
    words = sp.get_words(key_set)
    words2 = sp.get_words(key_set2)
    print(word_movers_distance(words, words2))


def full_wmd():
    ids = all_ids()  # Assuming this returns a list of user IDs
    n = len(ids)

    # Initialize an empty matrix for similarity scores
    similarity_matrix = np.zeros((n, n))

    sp = SentenceParser(os.path.join(os.getcwd(), "cleaned2.csv"))
    raw_data = read_compact_format()
    for i, uid in enumerate(all_ids()):
        for j, other in enumerate(all_ids()):
            df = get_user_by_platform_from_df(raw_data, uid, 1)
            df2 = get_user_by_platform_from_df(raw_data, other, 2)
            key_set = list(df["key"])
            key_set2 = list(df2["key"])
            words = sp.get_words(key_set)
            words2 = sp.get_words(key_set2)
            print(
                f"User {uid} and {other}:", float(word_movers_distance(words, words2))
            )
            # WMD measures dissimilarity
            similarity = float(word_movers_distance(words, words2))
            similarity_matrix[i, j] = similarity
    similarity_df = pd.DataFrame(similarity_matrix, index=ids, columns=ids)
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df, fmt=".2f")
    plt.title("WMD FI Heatmap")
    plt.xlabel("User ID")
    plt.ylabel("User ID")
    plt.savefig("WMD FI Heatmap.png")
    plt.show()


def simple_use_similarity():
    df = get_user_by_platform(1, 3)
    df2 = get_user_by_platform(4, 2)
    key_set = list(df["key"])
    key_set2 = list(df2["key"])
    sent1 = reconstruct_text(key_set)
    sent2 = reconstruct_text(key_set2)
    print(USE_similarity(sent1, sent2))


def simple_lsa_similarity():
    df = get_user_by_platform(1, 3)
    df2 = get_user_by_platform(1, 2)
    key_set = list(df["key"])
    key_set2 = list(df2["key"])
    sent1 = reconstruct_text(key_set)
    sent2 = reconstruct_text(key_set2)
    sim = lsa_similarity(sent1, sent2)
    print("Cosine Similarity (LSA):", sim)


def full_lsa():
    raw_data = read_compact_format()
    for uid in all_ids():
        for other in all_ids():
            df = get_df_by_user_id(raw_data, uid)
            df2 = get_df_by_user_id(raw_data, other)
            key_set = list(df["key"])
            sent1 = reconstruct_text(key_set)
            key_set2 = list(df2["key"])
            sent2 = reconstruct_text(key_set2)
            print(f"User {uid} and {other}:", lsa_similarity(sent1, sent2))


def s_test_cosine():
    sp = SentenceParser(os.path.join(os.getcwd(), "cleaned2.csv"))
    raw_data = read_compact_format()
    ids = all_ids()  # Assuming this returns a list of user IDs
    n = len(ids)

    # Initialize an empty matrix for similarity scores
    similarity_matrix = np.zeros((n, n))

    # SentenceTransformer model
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # Populate the matrix with similarity scores
    for i, uid in enumerate(ids):
        for j, other in enumerate(ids):
            df = get_user_by_platform_from_df(raw_data, uid, 2)
            df2 = get_user_by_platform_from_df(raw_data, other, 3)
            key_set = list(df["key"])
            key_set2 = list(df2["key"])
            s1 = sp.as_sentence(key_set)
            print(s1)
            s2 = sp.as_sentence(key_set2)
            embedding1 = model.encode(s1)
            embedding2 = model.encode(s2)
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            similarity_matrix[i, j] = similarity

    # Convert the matrix into a DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=ids, columns=ids)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df, fmt=".2f")
    plt.title("Sentence Cosine Similarity IT Heatmap")
    plt.xlabel("User ID")
    plt.ylabel("User ID")
    plt.savefig("Sentence Cosine Similarity IT Heatmap.png")
    plt.show()


def s_test_cosine_stylometric():
    sp = SentenceParser(os.path.join(os.getcwd(), "cleaned2.csv"))
    raw_data = read_compact_format()
    ids = all_ids()  # Assuming this returns a list of user IDs
    n = len(ids)

    # Initialize an empty matrix for similarity scores
    similarity_matrix = np.zeros((n, n))

    # Populate the matrix with similarity scores
    for i, uid in enumerate(ids):
        for j, other in enumerate(ids):
            df = get_user_by_platform_from_df(raw_data, uid, 1)
            df2 = get_user_by_platform_from_df(raw_data, other, 2)
            key_set = list(df["key"])
            key_set2 = list(df2["key"])
            s1 = sp.as_sentence(key_set)
            s2 = sp.as_sentence(key_set2)
            features1 = extract_stylometric_features(s1)
            features2 = extract_stylometric_features(s2)
            similarity = cosine_similarity([features1], [features2])[0][0]
            similarity_matrix[i, j] = similarity

    # Convert the matrix into a DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=ids, columns=ids)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df, fmt=".2f")
    plt.title("Sentence Cosine Similarity FI Heatmap")
    plt.xlabel("User ID")
    plt.ylabel("User ID")
    plt.savefig("Sentence Cosine Similarity FI Heatmap.png")
    plt.show()


def s_test_euclidean():
    sp = SentenceParser(os.path.join(os.getcwd(), "cleaned2.csv"))
    raw_data = read_compact_format()
    ids = all_ids()  # Assuming this returns a list of user IDs
    n = len(ids)

    # Initialize an empty matrix for similarity scores
    similarity_matrix = np.zeros((n, n))

    # SentenceTransformer model
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # Populate the matrix with similarity scores
    for i, uid in enumerate(ids):
        for j, other in enumerate(ids):
            df = get_user_by_platform_from_df(raw_data, uid, 1)
            df2 = get_user_by_platform_from_df(raw_data, other, 2)
            key_set = list(df["key"])
            key_set2 = list(df2["key"])
            s1 = sp.as_sentence(key_set)
            s2 = sp.as_sentence(key_set2)
            embedding1 = model.encode(s1)
            embedding2 = model.encode(s2)
            distance = euclidean_distances([embedding1], [embedding2])[0][0]
            similarity = 1 / (1 + distance)  # Convert distance to similarity
            similarity_matrix[i, j] = similarity

    # Convert the matrix into a DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=ids, columns=ids)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df, fmt=".2f")
    plt.title("Sentence Euclidean Similarity FI Heatmap")
    plt.xlabel("User ID")
    plt.ylabel("User ID")
    plt.savefig("Sentence Euclidean Similarity FI Heatmap.png")
    plt.show()


def s_test_manhattan():
    sp = SentenceParser(os.path.join(os.getcwd(), "cleaned2.csv"))
    raw_data = read_compact_format()
    ids = all_ids()  # Assuming this returns a list of user IDs
    n = len(ids)

    # Initialize an empty matrix for similarity scores
    similarity_matrix = np.zeros((n, n))

    # SentenceTransformer model
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # Populate the matrix with similarity scores
    for i, uid in enumerate(ids):
        for j, other in enumerate(ids):
            df = get_user_by_platform_from_df(raw_data, uid, 1)
            df2 = get_user_by_platform_from_df(raw_data, other, 2)
            key_set = list(df["key"])
            key_set2 = list(df2["key"])
            s1 = sp.as_sentence(key_set)
            s2 = sp.as_sentence(key_set2)
            embedding1 = model.encode(s1)
            embedding2 = model.encode(s2)
            distance = manhattan_distances([embedding1], [embedding2])[0][0]
            similarity = 1 / (1 + distance)  # Convert distance to similarity
            similarity_matrix[i, j] = similarity

    # Convert the matrix into a DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=ids, columns=ids)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df, fmt=".2f")
    plt.title("Sentence Stylometric Cosine Similarity FI Heatmap")
    plt.xlabel("User ID")
    plt.ylabel("User ID")
    plt.savefig("Sentence Manhattan Similarity FI Heatmap.png")
    plt.show()


def recon_test():
    sp = SentenceParser(os.path.join(os.getcwd(), "cleaned2.csv"))
    for i in tqdm(all_ids()):
        for j in range(1, 4):
            for k in range(1, 7):
                df = get_user_by_platform(i, j, k)
                if df.empty:
                    print(
                        f"Skipping User: {i}, platform: {map_platform_id_to_initial(j)}, session: {k}"
                    )
                    continue
            key_set = list(df["key"])
            text = sp.as_sentence(key_set)
            print(f"User: {i}, platform: {map_platform_id_to_initial(j)}, session: {k}")
            print(text)
            print()


full_wmd()
