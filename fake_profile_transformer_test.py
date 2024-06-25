import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
from core.sample_models.fake_profile_transformer import (
    NewKeystrokePairDataset,
    SimpleSiameseModel,
    TransformerSiameseModel,
    collate_fn,
    prepare_transformer_train_test_data,
)
from core.utils import create_kht_data_from_df, create_kit_data_from_df
from torch.utils.data import DataLoader
import torch.optim as optim

new_keystroke_data = pd.read_csv(
    os.path.join(os.getcwd(), "dataset", "test.csv"),
    dtype={
        "key": str,
        "press_time": np.float64,
        "release_time": np.float64,
        "platform_id": np.uint8,
        "session_id": np.uint8,
        "user_ids": np.uint8,
    },
)

kht_features = create_kht_data_from_df(new_keystroke_data)
# Generate KIT features for all types
kit_features = {}
for kit_type in range(1, 5):
    kit_features[kit_type] = create_kit_data_from_df(
        new_keystroke_data, kit_type, use_seperator=False
    )


def combine_features_into_df(df, kht_dict, kit_dict):
    df["kht"] = df.apply(
        lambda row: kht_dict[row["key"]][0] if row["key"] in kht_dict else 0, axis=1
    )

    for kit_type in range(1, 5):
        df[f"kit_{kit_type}"] = df.apply(
            lambda row, kt=kit_type: kit_dict[kt][row["key"] + row["key"]]
            if (row["key"] + row["key"]) in kit_dict[kt]
            else [0],
            axis=1,
        ).apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0)
    print(df)
    input()
    return df


new_keystroke_data = combine_features_into_df(
    new_keystroke_data, kht_features, kit_features
)
# Normalize the data
features_to_normalize = [
    "kht",
    "kit_1",
    "kit_2",
    "kit_3",
    "kit_4",
]
new_keystroke_data[features_to_normalize] = (
    new_keystroke_data[features_to_normalize]
    - new_keystroke_data[features_to_normalize].mean()
) / new_keystroke_data[features_to_normalize].std()
# Create the dataset and dataloader
new_keystroke_pair_dataset = NewKeystrokePairDataset(new_keystroke_data)
new_keystroke_pair_dataloader = DataLoader(
    new_keystroke_pair_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
)
train_data, val_data = prepare_transformer_train_test_data(new_keystroke_data)

train_dataset = NewKeystrokePairDataset(train_data)
val_dataset = NewKeystrokePairDataset(val_data)
train_dataloader = DataLoader(
    train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
)


def run_with_transformer_siamese():
    train_dataset = NewKeystrokePairDataset(train_data)
    train_dataloader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )

    # Hyperparameters
    input_dim = 5  # Adjusted for the additional features
    d_model = 64
    nhead = 8
    num_encoder_layers = 2
    dim_feedforward = 256

    # Instantiate the model
    model = TransformerSiameseModel(
        input_dim, d_model, nhead, num_encoder_layers, dim_feedforward
    )
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5  # Reduce the number of epochs for debugging
    for epoch in range(num_epochs):
        with tqdm(
            total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}"
        ) as pbar:
            model.train()
            for (seq1, seq2, user1, user2), labels in train_dataloader:
                optimizer.zero_grad()
                outputs = model(seq1, seq2)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        # Evaluate on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with tqdm(
                total=len(val_dataloader),
                desc=f"Validating Epoch {epoch+1}/{num_epochs}",
            ) as pbar:
                for (seq1, seq2, user1, user2), labels in val_dataloader:
                    outputs = model(seq1, seq2)
                    loss = criterion(outputs.squeeze(), labels)
                    val_loss += loss.item()
                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item())
        val_loss /= len(val_dataloader)
        print(f"Validation Loss: {val_loss:.4f}")

    # Compare every user against all other users in the validation set
    results = []
    unique_combinations = set()
    model.eval()
    with torch.no_grad():
        for (seq1, seq2, user1, user2), _ in tqdm(
            val_dataloader, desc="Comparing users"
        ):
            outputs = model(seq1, seq2)
            for u1, u2, score in zip(user1, user2, outputs):
                pair = tuple(sorted((u1.item(), u2.item())))
                if pair not in unique_combinations:
                    unique_combinations.add(pair)
                    results.append((u1.item(), u2.item(), score.item()))

    # Print the users and their similarity scores
    for u1, u2, score in results:
        print(f"User {u1} vs User {u2}: Similarity Score: {score:.4f}")


def run_with_simple_siamese():
    # Hyperparameters
    input_dim = 5  # Adjusted for the additional features
    hidden_dim = 32

    # Instantiate the model
    model = SimpleSiameseModel(input_dim, hidden_dim)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 30  # Reduce the number of epochs for faster training

    for epoch in range(num_epochs):
        for (seq1, seq2, user1, user2), labels in new_keystroke_pair_dataloader:
            optimizer.zero_grad()
            outputs = model(seq1, seq2)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Compare every user against all other users
    results = set()
    for (seq1, seq2, user1, user2), _ in new_keystroke_pair_dataloader:
        outputs = model(seq1, seq2)
        for u1, u2, score in zip(user1, user2, outputs):
            results.add((u1.item(), u2.item(), score.item()))

    # Print the users and their similarity scores
    for u1, u2, score in results:
        print(f"User {u1} vs User {u2}: Similarity Score: {score:.4f}")


run_with_transformer_siamese()
