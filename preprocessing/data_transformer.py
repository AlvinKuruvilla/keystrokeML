import os
import pandas as pd
from rich.progress import track


def is_sequence_match(current_seq, sequence):
    return current_seq == sequence


def get_daatset_file(path: str):
    df = pd.read_csv(path, header=None)
    df.columns = ["direction", "key", "time"]
    return df


def get_platform_id_and_user_id(filename: str):
    parts = filename.split("_")
    user_id = parts[1]
    platform_coefficent = parts[0]
    if platform_coefficent == "f":
        return (1, user_id)
    elif platform_coefficent == "i":
        return (2, user_id)
    elif platform_coefficent == "t":
        return (3, user_id)
    else:
        raise ValueError("invalid platform coefficent")


def find_pairs(data: pd.DataFrame, filename: str):
    platform_id, user_id = get_platform_id_and_user_id(filename)

    # Initialize a list to store the results
    result = []

    # Create a dictionary to store the last press event for each key
    last_press = {}

    # Iterate through the rows of the dataframe
    for _, row in track(data.iterrows()):
        key = row["key"]
        timestamp = row["time"]

        if row["direction"] == "P":
            # Record the press event
            last_press[key] = timestamp
        elif row["direction"] == "R" and key in last_press:
            # Record the release event and generate the result entry
            result.append([key, last_press[key], timestamp, platform_id, user_id])
            # Remove the key from the dictionary to handle multiple presses and releases
            del last_press[key]
    fields = ["key", "press_time", "release_time", "platform_id", "user_id"]
    df = pd.DataFrame(result, columns=fields)
    session_id = 1
    sequence = ["x", "x", "x", "x", "x", "y", "y", "y", "y", "y"]
    seq_len = len(sequence)
    current_seq = []

    # Iterate over the dataframe and update session_id
    for i in range(len(df)):
        # Add the current key to the sequence being tracked
        current_seq.append(df.loc[i, "key"].strip("'").lower())

        # If the sequence exceeds the required length, remove the oldest entry
        if len(current_seq) > seq_len:
            current_seq.pop(0)

        # Assign session_id to the current row
        df.loc[i, "session_id"] = session_id

        # Check if the current sequence matches the target sequence
        if current_seq == sequence:
            session_id += 1
            current_seq = []  # Reset the current sequence after a match
        print(current_seq)
    # Convert session_id to integer type
    df["session_id"] = df["session_id"].astype(int)
    return df


def get_all_pairs(path):
    files = os.listdir(path)
    data = []
    for file in files:
        df = get_daatset_file(os.path.join(os.getcwd(), path, file))
        data.append(find_pairs(df, file))
    pd.concat(data, axis=0).to_csv("attempt.csv", index=False)


get_all_pairs(os.path.join(os.getcwd(), "dataset", "indr"))
