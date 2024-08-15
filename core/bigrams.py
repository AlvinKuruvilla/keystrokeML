import os
from collections import Counter


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


if __name__ == "__main__":
    words = read_word_list()
    bigrams = sorted_bigrams_frequency(words)
