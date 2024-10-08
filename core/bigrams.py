import os
import pandas as pd
from collections import Counter
from functools import cache


# http://norvig.com/mayzner.html
def peter_norvig_bigrams():
    return [
        "TH",
        "HE",
        "IN",
        "ER",
        "AN",
        "RE",
        "ON",
        "AT",
        "EN",
        "ND",
        "TI",
        "ES",
        "OR",
        "TE",
        "OF",
        "ED",
        "IS",
        "IT",
        "AL",
        "AR",
        "ST",
        "TO",
        "NT",
        "NG",
        "SE",
        "HA",
        "AS",
        "OU",
        "IO",
        "LE",
        "VE",
        "CO",
        "ME",
        "DE",
        "HI",
        "RI",
        "RO",
        "IC",
        "NE",
        "EA",
        "RA",
        "CE",
        "LI",
        "CH",
        "LL",
        "BE",
        "MA",
        "SI",
        "OM",
        "UR",
    ]


# https://mathcenter.oxford.emory.edu/site/math125/englishLetterFreqs/
def oxford_bigrams():
    return [
        "th",
        "he",
        "in",
        "en",
        "nt",
        "re",
        "er",
        "an",
        "ti",
        "es",
        "on",
        "at",
        "se",
        "nd",
        "or",
        "ar",
        "al",
        "te",
        "co",
        "de",
        "to",
        "ra",
        "et",
        "ed",
        "it",
        "sa",
        "em",
        "ro",
    ]


def read_word_list():
    with open(os.path.join(os.getcwd(), "words_alpha.txt")) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def generate_bigrams(words):
    """Generate bigrams from a list of words and count their frequency."""
    bigrams = []
    for word in words:
        word = str(word)
        bigrams.extend([word[i : i + 2] for i in range(len(word) - 1)])
    return Counter(bigrams)


def sorted_bigrams_frequency(words):
    """Sort bigrams by frequency from highest to lowest."""
    bigram_counts = generate_bigrams(words)
    return sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)


def generate_unique_bigrams(word):
    bigrams = set()  # Using a set to store unique bigrams
    word = str(word)
    for i in range(len(word) - 1):
        bigram = str(word[i : i + 2])
        bigrams.add(bigram)
    return bigrams


@cache
def read_norvig_words(n=15):
    df = pd.read_csv("peter_norvig_words.txt", sep="\s+", header=None)
    # Extract the first column
    bigrams = []
    bigram_pairs = {}
    words = list(df[0])
    frequencies = list(df[1])
    for word in words:
        word = str(word)
        print(word)
        bigrams.extend(generate_unique_bigrams(word))
    unique_bigrams = set(bigrams)
    for bigram in unique_bigrams:
        for idx, word in enumerate(words):
            if bigram in str(word):
                if bigram in bigram_pairs:
                    bigram_pairs[bigram] += frequencies[idx]
                else:
                    bigram_pairs[bigram] = 0
    sorted_dict = dict(
        sorted(bigram_pairs.items(), key=lambda item: item[1], reverse=True)
    )
    return list(sorted_dict.keys())[0:n]


if __name__ == "__main__":
    words = read_word_list()
    bigrams = sorted_bigrams_frequency(words)
