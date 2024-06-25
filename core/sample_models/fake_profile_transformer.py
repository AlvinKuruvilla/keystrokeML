from itertools import combinations_with_replacement
import torch
import pandas as pd
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# Define the custom dataset for pairwise data
class NewKeystrokePairDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.sequences = (
            data.groupby(["user_ids", "session_id"])
            .apply(
                lambda x: x[
                    [
                        "kht",
                        "kit_1",
                        "kit_2",
                        "kit_3",
                        "kit_4",
                    ]
                ].values
            )
            .reset_index(drop=True)
        )
        self.user_ids = (
            data.groupby(["user_ids", "session_id"])
            .apply(lambda x: x["user_ids"].values[0])
            .reset_index(drop=True)
        )
        self.pairs, self.labels = self.create_pairs()

    def create_pairs(self):
        pairs = []
        labels = []
        for i, j in combinations_with_replacement(range(len(self.sequences)), 2):
            pairs.append(
                (
                    self.sequences[i],
                    self.sequences[j],
                    self.user_ids[i],
                    self.user_ids[j],
                )
            )
            labels.append(int(self.user_ids[i] == self.user_ids[j]))
        return pairs, labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seq1, seq2, user1, user2 = self.pairs[idx]
        label = self.labels[idx]
        return (
            torch.tensor(seq1, dtype=torch.float32),
            torch.tensor(seq2, dtype=torch.float32),
            user1,
            user2,
        ), torch.tensor(label, dtype=torch.float32)


def collate_fn(batch):
    seq1, seq2, user1, user2, labels = zip(
        *[(item[0][0], item[0][1], item[0][2], item[0][3], item[1]) for item in batch]
    )
    seq1_padded = pad_sequence(seq1, batch_first=True, padding_value=0)
    seq2_padded = pad_sequence(seq2, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    return (seq1_padded, seq2_padded, user1, user2), labels


# Define a transformer-based model
class TransformerSiameseModel(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        max_len=5000,
        dropout=0.1,
    ):
        super(TransformerSiameseModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_encoder_layers
        )

        self.fc1 = nn.Linear(d_model, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

        self.positional_encoding = self._generate_positional_encoding(d_model, max_len)

    def _generate_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, seq1, seq2):
        _, seq_len_1, _ = seq1.size()
        _, seq_len_2, _ = seq2.size()

        # Apply embedding
        seq1 = self.embedding(seq1)
        seq2 = self.embedding(seq2)

        # Add positional encoding
        seq1 = seq1 + self.positional_encoding[:, :seq_len_1, :].to(seq1.device)
        seq2 = seq2 + self.positional_encoding[:, :seq_len_2, :].to(seq2.device)

        # Transform sequences
        transformed_seq1 = self.transformer_encoder(seq1)
        transformed_seq2 = self.transformer_encoder(seq2)

        # Average over sequence length
        transformed_seq1 = transformed_seq1.mean(dim=1)
        transformed_seq2 = transformed_seq2.mean(dim=1)

        # Compute distance and classify
        distance = torch.abs(transformed_seq1 - transformed_seq2)
        x = torch.relu(self.fc1(distance))
        x = self.dropout(x)
        output = torch.sigmoid(self.fc2(x))
        return output


def prepare_transformer_train_test_data(df):
    ids = [num for num in range(1, 26) if num != 22]
    train = []
    test = []
    for user_id in ids:
        user_data = df[(df["user_ids"] == user_id)]
        split_index = len(user_data) // 2
        train.append(user_data.iloc[:split_index])
        test.append(user_data.iloc[split_index:])
    return (pd.concat(train), pd.concat(test))


# Define a simpler model
class SimpleSiameseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleSiameseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, seq1, seq2):
        seq1 = seq1.mean(dim=1)
        seq2 = seq2.mean(dim=1)
        distance = torch.abs(seq1 - seq2)
        x = torch.relu(self.fc1(distance))
        output = torch.sigmoid(self.fc2(x))
        return output
