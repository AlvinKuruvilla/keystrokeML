import numpy as np
import pandas as pd
from collections import defaultdict

from bigrams import read_word_list, sorted_bigrams_frequency


def sort_kit_data_by_values_length(data):
    return sorted(data.items(), key=lambda item: len(item[1]), reverse=True)


def compute_median_features(sorted_data):
    # Compute the median for the lists of the given sorted data and return as a dictionary
    return {key: np.median(values) for key, values in sorted_data.items()}


def create_kht_feature_matrix(data):
    matrix = defaultdict(list)

    for key in list(data.keys()):
        res = np.median(list(data[key]))
        # if res < 0:
        #     print("Encountered negative median")
        #     print(res)
        #     print(key)
        #     print(data[key])
        #     input()
        matrix[key].append(res)
    return pd.DataFrame.from_dict(matrix)


def most_common_features(data):
    words = read_word_list()
    all_bigrams = sorted_bigrams_frequency(words)

    # Filter and sort bigrams that are in the provided data defaultdict
    filtered_bigrams = [
        (bigram, data[bigram]) for bigram in all_bigrams if bigram in data
    ]
    filtered_bigrams.sort(key=lambda x: x[1], reverse=True)

    bigrams = filtered_bigrams[:30]
    res = defaultdict(list)
    for bigram in bigrams:
        res[bigram].append(dict(data[bigram]))
    return compute_median_features(res)


def least_common_features(data):
    words = read_word_list()
    all_bigrams = sorted_bigrams_frequency(words)

    # Filter and sort bigrams that are in the provided data defaultdict
    filtered_bigrams = [
        (bigram, data[bigram]) for bigram in all_bigrams if bigram in data
    ]
    filtered_bigrams.sort(key=lambda x: x[1])

    bigrams = filtered_bigrams[:30]
    res = defaultdict(list)
    for bigram in bigrams:
        res[bigram].append(dict(data[bigram]))
    return compute_median_features(res)


def combine_kit_features(dd1, dd2):
    combined_dd = defaultdict(list, dd1)  # Start with a copy of dd1
    for key, values in dd2.items():
        combined_dd[key].append(values)
    return combined_dd


def create_kit_feature_matrix(data):
    sorted_data = sort_kit_data_by_values_length(data)
    least_common = least_common_features(sorted_data)
    most_common = most_common_features(sorted_data)
    res = combine_kit_features(least_common, most_common)
    return pd.DataFrame.from_dict(res)


def combine_into_feature_matrix(kht_df, kit_df):
    return pd.concat([kht_df, kit_df], axis=1)
