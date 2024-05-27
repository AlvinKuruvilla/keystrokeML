import os
from collections import defaultdict
import numpy as np
import pandas as pd


def read_compact_format():
    df = pd.read_csv(
        os.path.join(os.getcwd(), "dataset", "cleaned2.csv"),
        dtype={
            "key": str,
            "press_time": np.float64,
            "release_time": np.float64,
            "platform_id": np.uint8,
            "session_id": np.uint8,
            "user_ids": np.uint8,
        },
    )
    return df


def create_kht_data_from_df(df):
    """
    Computes Key Hold Time (KHT) data from a given dataframe.

    Parameters:
    - df (pandas.DataFrame): A dataframe with columns "key", "press_time", and "release_time",
      where each row represents an instance of a key press and its associated press and release times.

    Returns:
    - dict: A dictionary where keys are individual key characters and values are lists containing
      computed KHT values (difference between the release time and press time) for each instance of the key.

    Note:
    KHT is defined as the difference between the release time and the press time for a given key instance.
    This function computes the KHT for each key in the dataframe and aggregates the results by key.
    """
    kht_dict = defaultdict(list)
    for i, row in df.iterrows():
        kht_dict[row["key"]].append(row["release_time"] - row["press_time"])
    return kht_dict


def create_kit_data_from_df(df, kit_feature_type, use_seperator: bool = True):
    """
    Computes Key Interval Time (KIT) data from a given dataframe based on a specified feature type.

    Parameters:
    - df (pandas.DataFrame): A dataframe with columns "key", "press_time", and "release_time",
      where each row represents an instance of a key press and its associated press and release times.

    - kit_feature_type (int): Specifies the type of KIT feature to compute. The valid values are:
      1: Time between release of the first key and press of the second key.
      2: Time between release of the first key and release of the second key.
      3: Time between press of the first key and press of the second key.
      4: Time between press of the first key and release of the second key.

    Returns:
    - dict: A dictionary where keys are pairs of consecutive key characters and values are lists containing
      computed KIT values based on the specified feature type for each instance of the key pair.

    Note:
    This function computes the KIT for each pair of consecutive keys in the dataframe and aggregates
    the results by key pair. The method for computing the KIT is determined by the `kit_feature_type` parameter.
    """
    kit_dict = defaultdict(list)
    if df.empty:
        # print("dig deeper: dataframe is empty!")
        return kit_dict
    num_rows = len(df.index)
    for i in range(num_rows):
        if i < num_rows - 1:
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]
            if use_seperator:
                key = current_row["key"] + "|*" + next_row["key"]
            else:
                key = current_row["key"] + next_row["key"]
            initial_press = float(current_row["press_time"])
            second_press = float(next_row["press_time"])
            initial_release = float(current_row["release_time"])
            second_release = float(next_row["release_time"])
            if kit_feature_type == 1:
                kit_dict[key].append(second_press - initial_release)
            elif kit_feature_type == 2:
                kit_dict[key].append(second_release - initial_release)
            elif kit_feature_type == 3:
                kit_dict[key].append(second_press - initial_press)
            elif kit_feature_type == 4:
                kit_dict[key].append(second_release - initial_press)
    return kit_dict


def map_platform_id_to_initial(platform_id: int):
    platform_mapping = {1: "f", 2: "i", 3: "t"}

    if platform_id not in platform_mapping:
        raise ValueError(f"Bad platform_id: {platform_id}")

    return platform_mapping[platform_id]
