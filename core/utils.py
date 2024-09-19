import os
import re
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from rich.progress import track
from functools import cache


def clean_string(s):
    # Remove extraneous single quotes and retain the actual content
    cleaned = re.sub(r"'\s*|\s*'", "", s)

    # Remove any extra spaces
    return cleaned.strip()


def clean_strings(strings):
    cleaned_strings = []

    for s in strings:
        cleaned = clean_string(s)
        # Append the cleaned string to the result list
        cleaned_strings.append(cleaned)

    return cleaned_strings


@cache
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


def read_compact_format_with_gender():
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
    ids = list(df["user_ids"])
    genders = []
    with open(os.path.join(os.getcwd(), "genders.json"), "r") as f:
        data = json.load(f)
    for uid in ids:
        genders.append(data[str(uid)])
    df["genders"] = genders
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
    # print("Feature type:", kit_feature_type)
    if kit_feature_type not in range(1, 5):
        raise ValueError(f"Bad kit feature type {kit_feature_type}")
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
                key = clean_string(current_row["key"] + next_row["key"])
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


def create_kit_features_combined(df):
    kit_feature_types = [1, 2, 3, 4]
    kit_features_combined = defaultdict(list)

    for kit_feature_type in kit_feature_types:
        kit_features = create_kit_data_from_df(df, kit_feature_type, False)
        for key, values in kit_features.items():
            combined_key = f"{key}_type_{kit_feature_type}"
            kit_features_combined[combined_key] = values

    return kit_features_combined


def get_df_by_user_id(df: pd.DataFrame, user_id):
    return df[df["user_ids"] == user_id]


def get_user_by_platform(user_id, platform_id, session_id=None):
    """
    Retrieve data for a given user and platform, with an optional session_id filter.

    Parameters:
    - user_id (int): Identifier for the user.
    - platform_id (int or list[int]): Identifier for the platform.
      If provided as a list, it should contain two integers specifying
      an inclusive range to search between.
    - session_id (int or list[int], optional): Identifier for the session.
      If provided as a list, it can either specify an inclusive range with
      two integers or provide multiple session IDs to filter by.

    Returns:
    - DataFrame: Filtered data matching the given criteria.

    Notes:
    - When providing a list for platform_id or session_id to specify a range,
      the order of the two integers does not matter.
    - When providing a list with more than two integers for session_id,
      it will filter by those exact session IDs.

    Raises:
    - AssertionError: If platform_id or session_id list does not follow the expected format.

    Examples:
    >>> df = get_user_by_platform(123, 1)
    >>> df = get_user_by_platform(123, [1, 5])
    >>> df = get_user_by_platform(123, 1, [2, 6])
    >>> df = get_user_by_platform(123, 1, [2, 3, 4])

    """
    # Get all of the data for a user amd platform with am optional session_id

    # print(f"user_id:{user_id}", end=" | ")
    df = read_compact_format()
    if session_id is None:
        if isinstance(platform_id, list):
            # Should only contain an inclusive range of the starting id and ending id
            assert len(platform_id) == 2
            if platform_id[0] < platform_id[1]:
                return df[
                    (df["user_ids"] == user_id)
                    & (df["platform_id"].between(platform_id[0], platform_id[1]))
                ]
            else:
                return df[
                    (df["user_ids"] == user_id)
                    & (df["platform_id"].between(platform_id[1], platform_id[0]))
                ]

        return df[(df["user_ids"] == user_id) & (df["platform_id"] == platform_id)]
    if isinstance(session_id, list):
        # Should only contain an inclusive range of the starting id and ending id
        if len(session_id) == 2:
            return df[
                (df["user_ids"] == user_id)
                & (df["platform_id"] == platform_id)
                & (df["session_id"].between(session_id[0], session_id[1]))
            ]
        elif len(session_id) > 2:
            test = df[
                (df["user_ids"] == user_id)
                & (df["platform_id"] == platform_id)
                & (df["session_id"].isin(session_id))
            ]
            # print(session_id)
            # print(test["session_id"].unique())
            # input()
            return df[
                (df["user_ids"] == user_id)
                & (df["platform_id"] == platform_id)
                & (df["session_id"].isin(session_id))
            ]

    return df[
        (df["user_ids"] == user_id)
        & (df["platform_id"] == platform_id)
        & (df["session_id"] == session_id)
    ]


def get_user_by_platform_from_df(df, user_id, platform_id, session_id=None):
    if session_id is None:
        if isinstance(platform_id, list):
            # Should only contain an inclusive range of the starting id and ending id
            assert len(platform_id) == 2
            if platform_id[0] < platform_id[1]:
                return df[
                    (df["user_ids"] == user_id)
                    & (df["platform_id"].between(platform_id[0], platform_id[1]))
                ]
            else:
                return df[
                    (df["user_ids"] == user_id)
                    & (df["platform_id"].between(platform_id[1], platform_id[0]))
                ]

        return df[(df["user_ids"] == user_id) & (df["platform_id"] == platform_id)]
    if isinstance(session_id, list):
        # Should only contain an inclusive range of the starting id and ending id
        if len(session_id) == 2:
            return df[
                (df["user_ids"] == user_id)
                & (df["platform_id"] == platform_id)
                & (df["session_id"].between(session_id[0], session_id[1]))
            ]
        elif len(session_id) > 2:
            test = df[
                (df["user_ids"] == user_id)
                & (df["platform_id"] == platform_id)
                & (df["session_id"].isin(session_id))
            ]
            # print(session_id)
            # print(test["session_id"].unique())
            # input()
            return df[
                (df["user_ids"] == user_id)
                & (df["platform_id"] == platform_id)
                & (df["session_id"].isin(session_id))
            ]

    return df[
        (df["user_ids"] == user_id)
        & (df["platform_id"] == platform_id)
        & (df["session_id"] == session_id)
    ]


def all_ids():
    return [num for num in range(1, 26) if num != 22]


def create_feature_dataframe(df, kht_func, kit_func, feature_types):
    """
    Create a dataframe with the features using KHT and KIT data.

    Parameters:
    - df (pandas.DataFrame): Input dataframe with columns "key", "press_time", and "release_time".
    - kht_func (function): Function to compute Key Hold Time (KHT) data.
    - kit_func (function): Function to compute Key Interval Time (KIT) data.
    - feature_types (list of int): List of KIT feature types to compute.

    Returns:
    - pandas.DataFrame: Dataframe with computed features.
    """
    feature_rows = []

    for i in range(len(df) - 1):
        # Extract the relevant part of the dataframe for the current sample
        sample_df = df.iloc[i : i + 2]

        # Compute KHT data for the sample
        kht_data = kht_func(sample_df)

        # Compute KIT data for the sample
        kit_data = {}
        for feature_type in feature_types:
            kit_data.update(kit_func(sample_df, feature_type))

        # Flatten the KHT and KIT data into a list of features for the current sample
        sample_features = []
        for key, values in kht_data.items():
            sample_features.extend(values)
        for key, values in kit_data.items():
            sample_features.extend(values)

        # Append the sample's features to the list of feature rows
        feature_rows.append(sample_features)

    # Determine the maximum number of features
    max_features = max(len(features) for features in feature_rows)

    # Create a dataframe with the features
    feature_columns = [f"tf{i}" for i in range(max_features)]
    feature_df = pd.DataFrame(feature_rows, columns=feature_columns)

    return feature_df


def get_features_dataframe(df: pd.DataFrame):
    feature_types = [1, 2, 3, 4]  # Specify the KIT feature types to compute

    return create_feature_dataframe(
        df, create_kht_data_from_df, create_kit_data_from_df, feature_types
    )
