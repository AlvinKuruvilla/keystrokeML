import os
import pandas as pd
import json
from rich.progress import track
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from core.utils import (
    all_ids,
    read_compact_format_with_gender,
    get_df_by_user_id,
    get_features_dataframe,
)

holder = []
data = read_compact_format_with_gender()
for uid in track(all_ids()):
    df_sample = get_df_by_user_id(data, 1)
    df = get_features_dataframe(df_sample)
    df["user_ids"] = uid
    with open(os.path.join(os.getcwd(), "genders.json"), "r") as f:
        gender_data = json.load(f)

    df["gender"] = gender_data[str(uid)]
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1, "Other": 2})

    # print(df)
    holder.append(df)
features_df = pd.concat(holder, axis=0)
# TODO: We are using all of the data, we should only the first few sessions for training and then the remainder to test
X = features_df.drop(["user_ids", "gender"], axis=1)
y = features_df["gender"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a RandomForestClassifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=["Male", "Female", "Other"])

print(report)
