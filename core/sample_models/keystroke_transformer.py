import os
import pandas as pd
from sklearn.base import TransformerMixin


class KeystrokeFeatureExtractor(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        grouped = X.groupby(["user_id", "seq_id"])

        for (user_id, seq_id), group in grouped:
            press_durations = group["release_time"] - group["press_time"]
            inter_key_delays = group["press_time"].iloc[1:].reset_index(
                drop=True
            ) - group["release_time"].iloc[:-1].reset_index(drop=True)
            features.append(
                [
                    user_id,
                    seq_id,
                    press_durations.mean(),
                    press_durations.std(),
                    inter_key_delays.mean(),
                    inter_key_delays.std(),
                ]
            )

        return pd.DataFrame(
            features,
            columns=[
                "user_id",
                "seq_id",
                "mean_press_duration",
                "std_press_duration",
                "mean_inter_key_delay",
                "std_inter_key_delay",
            ],
        )


def test_simple_transformer():
    # Load the dataset
    keystroke_data = pd.read_csv(
        os.path.join(os.getcwd(), "dataset", "synthetic_keystroke_data.csv"),
    )

    # Apply the transformer to the dataset
    extractor = KeystrokeFeatureExtractor()
    features = extractor.transform(keystroke_data)

    # Display the extracted features
    print(features.head())
