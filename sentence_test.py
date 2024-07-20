import os

from core.lexical_features import (
    USE_similarity,
    bert_similarity,
    lsa_similarity,
    mtldo,
    word_movers_distance,
)
from core.sentence_parser import SentenceParser, reconstruct_text
from core.utils import (
    all_ids,
    get_df_by_user_id,
    get_user_by_platform,
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
    sp = SentenceParser(os.path.join(os.getcwd(), "cleaned2.csv"))
    raw_data = read_compact_format()
    for uid in all_ids():
        for other in all_ids():
            df = get_df_by_user_id(raw_data, uid)
            df2 = get_df_by_user_id(raw_data, other)
            key_set = list(df["key"])
            key_set2 = list(df2["key"])
            words = sp.get_words(key_set)
            words2 = sp.get_words(key_set2)
            print(
                f"User {uid} and {other}:", float(word_movers_distance(words, words2))
            )


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
