import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Load the dataset
data = pd.read_csv(
    os.path.join(os.getcwd(), "dataset", "synthetic_keystroke_data.csv"),
)

# Extract temporal features
data["hold_latency"] = data["release_time"] - data["press_time"]
data["press_latency"] = data.groupby("seq_id")["press_time"].diff().fillna(0)
data["release_latency"] = data.groupby("seq_id")["release_time"].diff().fillna(0)
data["inner_key_latency"] = (
    data.groupby("seq_id")["press_time"].shift(-1) - data["release_time"]
)
data["outer_key_latency"] = data["press_time"] - data.groupby("seq_id")[
    "release_time"
].shift(1)

# Replace NaN values created by shift with 0
data.fillna(0, inplace=True)

# Create feature vectors for each sequence (grouped by 'seq_id')
features = data.groupby("seq_id").apply(
    lambda x: x[
        [
            "hold_latency",
            "press_latency",
            "release_latency",
            "inner_key_latency",
            "outer_key_latency",
        ]
    ].values
)
labels = data.groupby("seq_id")["user_id"].first().values

# Convert features to a suitable format for the model (padding sequences to the same length)
max_seq_length = max(map(len, features))
feature_dim = features[0].shape[1]
padded_features = np.zeros((len(features), max_seq_length, feature_dim))

for i, seq in enumerate(features):
    padded_features[i, : len(seq), :] = seq

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


class KeystrokeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


dataset = KeystrokeDataset(padded_features, encoded_labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, max_seq_length, hidden_dim=768):
        super(TransformerEncoder, self).__init__()
        self.max_seq_length = max_seq_length
        self.linear = nn.Linear(
            input_dim, hidden_dim
        )  # Adjust input dimension for transformer
        config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=max_seq_length,
        )
        self.bert = BertModel(config)

    def forward(self, x):
        x = self.linear(x)
        attention_mask = torch.ones(x.size()[:-1], dtype=torch.long).to(
            x.device
        )  # All ones mask
        outputs = self.bert(inputs_embeds=x, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)
        return pooled_output


class BiEncoder(nn.Module):
    def __init__(self, input_dim, max_seq_length, hidden_dim=768):
        super(BiEncoder, self).__init__()
        self.encoderA = TransformerEncoder(input_dim, max_seq_length, hidden_dim)
        self.encoderB = TransformerEncoder(input_dim, max_seq_length, hidden_dim)

    def forward(self, features_A, features_B):
        embedding_A = self.encoderA(features_A)
        embedding_B = self.encoderB(features_B)
        return embedding_A, embedding_B


def triplet_loss(anchor, positive, negative, margin=1.0):
    positive_distance = F.pairwise_distance(anchor, positive)
    negative_distance = F.pairwise_distance(anchor, negative)
    loss = F.relu(positive_distance - negative_distance + margin)
    return loss.mean()


def train_model(model, dataloader, optimizer, margin=1.0, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch in dataloader:
            features, labels = batch
            optimizer.zero_grad()

            half_size = features.size(0) // 2
            features_A, features_B = features[:half_size], features[half_size:]
            labels_A, labels_B = labels[:half_size], labels[half_size:]

            embedding_A, embedding_B = model(features_A, features_B)

            # Generate all possible triplets
            triplet_losses = []
            for i in range(embedding_A.size(0)):
                anchor = embedding_A[i]
                positive = embedding_B[i]
                for j in range(embedding_B.size(0)):
                    if i != j:
                        negative = embedding_B[j]
                        loss = triplet_loss(
                            anchor.unsqueeze(0),
                            positive.unsqueeze(0),
                            negative.unsqueeze(0),
                            margin,
                        )
                        triplet_losses.append(loss)

            if triplet_losses:
                total_batch_loss = torch.stack(triplet_losses).mean()
                total_batch_loss.backward()
                optimizer.step()
                total_loss += total_batch_loss.item()

            # Calculate accuracy for the batch
            positive_distances = F.pairwise_distance(embedding_A, embedding_B)
            negative_distances = torch.stack(
                [
                    F.pairwise_distance(
                        embedding_A[i].unsqueeze(0), embedding_B[j].unsqueeze(0)
                    )
                    for i in range(embedding_A.size(0))
                    for j in range(embedding_B.size(0))
                    if i != j
                ]
            )

            positive_predictions = (positive_distances < margin).float()
            negative_predictions = (negative_distances >= margin).float()
            correct_predictions += (
                positive_predictions.sum() + negative_predictions.sum()
            ).item()
            total_samples += positive_predictions.size(0) + negative_predictions.size(0)

        accuracy = correct_predictions / total_samples
        avg_loss = total_loss / len(dataloader)

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )


# Initialize model, optimizer
input_dim = 5  # Number of temporal features
hidden_dim = 768  # Hidden dimension for the transformer model
max_seq_length = padded_features.shape[
    1
]  # Ensure this matches your input data sequence length
model = BiEncoder(input_dim, max_seq_length, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Train the model
train_model(model, dataloader, optimizer, margin=1.0, epochs=3)
