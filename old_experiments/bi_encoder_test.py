import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from core.sample_models.fake_profile_bi_encoder import BiEncoder, KeystrokeDataset
from core.utils import create_kht_data_from_df, create_kit_data_from_df

# Initialize the model, loss function, and optimizer
embed_size = 128
heads = 8
depth = 6
model = BiEncoder(embed_size, heads, depth)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
data = pd.read_csv(
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
kht_features = create_kht_data_from_df(data)
kht_df = pd.DataFrame.from_dict(kht_features, orient="index").T

# Extract KIT features for all four types
kit_features_type1 = create_kit_data_from_df(data, kit_feature_type=1)
kit_features_type2 = create_kit_data_from_df(data, kit_feature_type=2)
kit_features_type3 = create_kit_data_from_df(data, kit_feature_type=3)
kit_features_type4 = create_kit_data_from_df(data, kit_feature_type=4)

kit_df_type1 = pd.DataFrame.from_dict(kit_features_type1, orient="index").T
kit_df_type2 = pd.DataFrame.from_dict(kit_features_type2, orient="index").T
kit_df_type3 = pd.DataFrame.from_dict(kit_features_type3, orient="index").T
kit_df_type4 = pd.DataFrame.from_dict(kit_features_type4, orient="index").T

# Merge KHT and KIT dataframes with the original data
features_df = pd.concat(
    [kht_df, kit_df_type1, kit_df_type2, kit_df_type3, kit_df_type4], axis=1
).fillna(0)
labels = data["user_ids"]

# Combine features and labels
processed_data = pd.concat([features_df, labels], axis=1)
# Initialize the model, loss function, and optimizer
embed_size = 128
heads = 8
depth = 6
model = BiEncoder(embed_size, heads, depth)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)

# Create DataLoader
train_dataset = KeystrokeDataset(train_data)
test_dataset = KeystrokeDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        x, y = batch
        optimizer.zero_grad()
        y_hat = model(x).squeeze()
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Validate the model
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        y_hat = model(x).squeeze()
        preds = torch.round(torch.sigmoid(y_hat))
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

accuracy = accuracy_score(all_labels, all_preds)
print(accuracy)
