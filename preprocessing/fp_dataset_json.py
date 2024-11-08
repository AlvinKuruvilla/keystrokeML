import numpy as np
import pandas as pd
import os
import json


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


def get_triplet(df):
    formatted_data_list = []
    for _, row in df.iterrows():
        key = row["key"].strip("'")  # Remove single quotes around the key if any
        press_time = float(row["press_time"])
        release_time = float(row["release_time"])
        # Format the data as specified
        formatted_data = [
            key,  # Key value
            "KD",
            str(press_time),
        ]
        formatted_data_list.append(formatted_data)
        formatted_data = [
            key,  # Key value
            "KU",
            str(release_time),
        ]
        formatted_data_list.append(formatted_data)
    return formatted_data_list


ids = all_ids()
data = {}
for user_id in ids:
    for platform_id in range(1, 4):
        df = get_user_by_platform(user_id, platform_id)
        keyboard_data = {}
        json_key = str(user_id) + "_" + str(platform_id)
        data[json_key] = keyboard_data
        keyboard_data["keyboard_data"] = get_triplet(df)
        pass
print(data)
with open("fp_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
