from itertools import combinations_with_replacement
import torch
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
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward):
        super(TransformerSiameseModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_encoder_layers
        )
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, seq1, seq2):
        seq1 = self.embedding(seq1)
        seq2 = self.embedding(seq2)
        seq1 = seq1.permute(
            1, 0, 2
        )  # Transpose for transformer input (seq_len, batch, feature)
        seq2 = seq2.permute(
            1, 0, 2
        )  # Transpose for transformer input (seq_len, batch, feature)
        transformed_seq1 = self.transformer_encoder(seq1)
        transformed_seq2 = self.transformer_encoder(seq2)
        transformed_seq1 = transformed_seq1.mean(dim=0)  # Average over sequence length
        transformed_seq2 = transformed_seq2.mean(dim=0)  # Average over sequence length
        distance = torch.abs(transformed_seq1 - transformed_seq2)
        x = torch.relu(self.fc1(distance))
        output = torch.sigmoid(self.fc2(x))
        return output


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
