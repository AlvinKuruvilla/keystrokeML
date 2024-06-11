import numpy as np
import pandas as pd
from rich.progress import track

# Parameters for synthetic data generation
num_users = 1000
num_sequences_per_user = 15
sequence_length = 70  # Number of keystrokes per sequence


# Function to generate a single keystroke sequence
def generate_keystroke_sequence(sequence_length):
    keycodes = np.random.randint(0, 255, sequence_length)
    press_times = np.cumsum(np.random.uniform(0.05, 0.3, sequence_length))
    release_times = press_times + np.random.uniform(0.05, 0.1, sequence_length)
    return keycodes, press_times, release_times


def extract_features(df):
    df["HL"] = df["release_time"] - df["press_time"]
    df["PL"] = df["press_time"].diff().fillna(0)
    df["RL"] = df["release_time"].diff().fillna(0)
    df["IL"] = df["press_time"] - df["release_time"].shift(1).fillna(0)
    features = df[["keycode", "HL", "PL", "RL", "IL"]].values
    return features


# Generate synthetic dataset
if __name__ == "__main__":
    data = []
    for user_id in track(range(num_users)):
        for seq_id in range(num_sequences_per_user):
            keycodes, press_times, release_times = generate_keystroke_sequence(
                sequence_length
            )
            for k, p, r in zip(keycodes, press_times, release_times):
                data.append([user_id, seq_id, k, p, r])

    df = pd.DataFrame(
        data, columns=["user_id", "seq_id", "keycode", "press_time", "release_time"]
    )
    df.to_csv("dataset/synthetic_keystroke_data.csv", index=False)

    print("Synthetic dataset created and saved as 'synthetic_keystroke_data.csv'.")
