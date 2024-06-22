import os
from collections import defaultdict
import numpy as np
import pandas as pd
from rich.progress import track


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


# TODO: Use the actual Unicode keycode eventually
# TODO: This is going to be a bad solution when we have to do KIT
def key_to_keycode(keys, key, use_kit):
    if use_kit:
        data = key.split("|*")
        return str(keys.index(data[0])) + str(keys.index(data[1]))
    return keys.index(key)


def make_into_timeseries_df(use_kit, kit_index=None):
    data = []
    df = read_compact_format()
    all_keys = list(set(df["key"]))
    all_keys.sort()
    for i in track(range(1, 26)):
        if i == 22:
            continue
        for j in range(1, 7):
            for k in range(1, 4):
                print("User ID:", i)
                print("Platform ID:", k)
                print("Session ID:", j)
                df = read_compact_format()
                rem = df[
                    (df["user_ids"] == i)
                    & (df["session_id"] == j)
                    & (df["platform_id"] == k)
                ]
                if rem.empty:
                    print(
                        f"Skipping user_id: {i} and platform id: {map_platform_id_to_initial(k)} and session_id: {j}"
                    )
                    continue
                if use_kit:
                    if kit_index is not None:
                        kit = create_kit_data_from_df(rem, kit_index)
                        for key, timings in kit.items():
                            entry = {
                                "user_id": i,
                                "session_id": j,
                                "platform_id": k,
                                "key": key_to_keycode(all_keys, key, True),
                                "kht_value": np.median(list(timings)),
                            }
                            data.append(entry)
                    else:
                        raise ValueError()
                else:
                    kht = create_kht_data_from_df(rem)
                    for key, timings in kht.items():
                        entry = {
                            "user_id": i,
                            "session_id": j,
                            "platform_id": k,
                            "key": key_to_keycode(all_keys, key, False),
                            "kht_value": np.median(list(timings)),
                        }
                        data.append(entry)
    return pd.DataFrame(data, columns=list(data[0].keys()))


def transformer_df_into_feature_vectors(data):
    # Extract feature columns and convert to numpy array
    return data.iloc[:, 2:].values
