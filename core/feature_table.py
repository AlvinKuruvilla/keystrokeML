import enum
import random
import statistics
from functools import cache
import pandas as pd
import numpy as np
from collections import defaultdict
from core.bigrams import oxford_bigrams, read_word_list, sorted_bigrams_frequency
from core.deft import find_avg_deft_for_deft_distance_and_kit_feature, flatten_list
from core.utils import clean_string, create_kit_data_from_df, read_compact_format


class CKP_SOURCE(enum.Enum):
    FAKE_PROFILE_DATASET = 0
    ALPHA_WORDS = 1
    OXFORD_EMORY = 2
    NORVIG = 3


# TODO: We need a more algorithmic approach to finding the smallest possible set of common keypairs which does not give any empty sub-arrays
# NOTE: The n=10 is temporary
@cache
def most_common_kepairs(n=5):
    freq = {}
    df = read_compact_format()
    kit1 = create_kit_data_from_df(df, 1, False)
    k = list(kit1.keys())
    for key in k:
        if key not in freq:
            freq[key] = len(kit1[key])
    # Sort the dictionary items by frequency in descending order
    sorted_items = sorted(freq.items(), key=lambda item: item[1], reverse=True)
    # Extract the top 'n' keys from the sorted list
    return [key for key, _ in sorted_items[:n]]


def alpha_word_bigrams(m=15):
    count = 0
    bigrams = []
    words = read_word_list()
    bigram_frequencies = sorted_bigrams_frequency(words)
    for bigram, _ in bigram_frequencies:
        if count == m:
            return bigrams
        bigrams.append(bigram)
        count += 1


def get_ckps(ckp_source: CKP_SOURCE):
    if ckp_source == CKP_SOURCE.FAKE_PROFILE_DATASET:
        return most_common_kepairs()
    elif ckp_source == CKP_SOURCE.ALPHA_WORDS:
        return alpha_word_bigrams()
    elif ckp_source == CKP_SOURCE.OXFORD_EMORY:
        return oxford_bigrams()
    elif ckp_source == CKP_SOURCE.NORVIG:
        return read_norvig_words()


class KeystrokeFeatureTable:
    def __init__(self) -> None:
        self.inner = defaultdict(list)

    # NOTE: We should not use the most common keypairs for the deft features because they rely on the distances between keys rather
    # than timing differences so all users may show up the same but I have to check
    def find_kit_from_most_common_keypairs(self, df, ckp_source: CKP_SOURCE):
        common_keypairs = get_ckps(ckp_source)
        for ckp in common_keypairs:
            for i in range(1, 5):
                # print(
                #     ckp
                #     in list(create_kit_data_from_df(df, i, use_seperator=False).keys())
                # )
                # print(ckp)
                # print(list(create_kit_data_from_df(df, i, use_seperator=False).keys()))
                # input()
                self.inner[ckp] = list(
                    create_kit_data_from_df(df, i, use_seperator=False)[
                        clean_string(ckp)
                    ]
                )

    def find_deft_for_df(self, df):
        for kit_feature in range(1, 5):
            for deft_val in range(0, 4):
                res = find_avg_deft_for_deft_distance_and_kit_feature(
                    df, deft_val, kit_feature
                )
                key_name = f"kit_{str(kit_feature)}_deft_{str(deft_val)}"
                self.inner[key_name] = res

    def get_raw(self):
        return self.inner

    def add_user_platform_session_identifiers(self, user_id, platform, session_id):
        self.inner["user_id"] = user_id
        self.inner["platform_id"] = platform
        if session_id is not None:
            self.inner["session_id"] = session_id

    def as_df(self):
        data = {key: [] for key in self.inner}
        for key, values in self.inner.items():
            if isinstance(values, list):
                data[key].append(values)
            else:
                data[key].append([values])

        return pd.DataFrame(data)


def only_user_id(df: pd.DataFrame, uid):
    return df[df["user_id"].apply(lambda x: x[0]) == uid]


def only_platform_id(df: pd.DataFrame, platform_id):
    return df[df["platform_id"].apply(lambda x: x[0]) == platform_id]


def fill_empty_row_values(df: pd.DataFrame, ckps):
    cols = df.columns
    diffs = []
    for col in cols:
        if col in ckps:
            flat_data = flatten_list(list(df[col]))
            data = statistics.mean(flatten_list(list(df[col])))
            print(data)
            for element in flat_data:
                diffs.append(element - data)
            replacement_value = random.uniform(min(diffs), max(diffs))
            df[col] = df[col].apply(lambda x: [replacement_value] if x == [] else x)
    return df


def compute_fixed_feature_values(lst):
    if len(lst) < 2:
        # This is just a way to make all of the KIT feature columns have the same length at the end
        # we can revert this back to just return the single element if we want to
        return [lst[0], lst[0], lst[0], lst[0], lst[0]]
        # return lst

    # Convert to numpy array for convenience
    arr = np.array(lst)
    # Return the statistics as a list
    return [np.min(arr), np.max(arr), np.median(arr), np.mean(arr), np.std(arr)]


def flatten_kit_feature_columns(df: pd.DataFrame, ckps):
    cols = df.columns
    for col in cols:
        if col in ckps:
            df[col] = df[col].apply(
                lambda x: compute_fixed_feature_values(x)
                if isinstance(x, list) and len(x) >= 2
                else x
            )
    return df


def drop_empty_list_columns(df):
    # Identify columns where all values are empty lists
    columns_to_drop = [
        col for col in df.columns if df[col].apply(lambda x: x == []).all()
    ]

    # Drop these columns
    df_dropped = df.drop(columns=columns_to_drop)

    return df_dropped


def read_norvig_words(n=15):
    df = pd.read_csv("peter_norvig_words.txt", delim_whitespace=True, header=None)
    # Extract the first column
    words = list(df[0])
    count = 0
    bigrams = []
    bigram_frequencies = sorted_bigrams_frequency(words)
    for bigram, _ in bigram_frequencies:
        if count == n:
            return bigrams
        bigrams.append(bigram)
        count += 1