import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


# Define the custom dataset for pairwise data
class KeystrokePairDataset(Dataset):
    def __init__(self, data, num_pairs=1000):
        self.data = data
        self.sequences = (
            data.groupby(["user_id", "seq_id"])
            .apply(lambda x: x[["keycode", "press_time", "release_time"]].values)
            .reset_index(drop=True)
        )
        self.user_ids = (
            data.groupby(["user_id", "seq_id"])
            .apply(lambda x: x["user_id"].values[0])
            .reset_index(drop=True)
        )
        self.pairs, self.labels = self.create_pairs(num_pairs)

    def create_pairs(self, num_pairs):
        pairs = []
        labels = []
        count = 0
        for i in range(len(self.sequences)):
            for j in range(i + 1, len(self.sequences)):
                pairs.append(
                    (
                        self.sequences[i],
                        self.sequences[j],
                        self.user_ids[i],
                        self.user_ids[j],
                    )
                )
                labels.append(int(self.user_ids[i] == self.user_ids[j]))
                count += 1
                if count >= num_pairs:
                    return pairs, labels
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


# Reduce the dataset size for demonstration
keystroke_data_small = pd.read_csv(
    os.path.join(os.getcwd(), "dataset", "synthetic_keystroke_data.csv"),
)

keystroke_pair_dataset = KeystrokePairDataset(keystroke_data_small, num_pairs=100)
keystroke_pair_dataloader = DataLoader(
    keystroke_pair_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
)


# Define a very simple model
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


# Hyperparameters
input_dim = 3
hidden_dim = 16

# Instantiate the model
model = SimpleSiameseModel(input_dim, hidden_dim)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Reduce the number of epochs for demonstration

for epoch in range(num_epochs):
    for (seq1, seq2, user1, user2), labels in keystroke_pair_dataloader:
        optimizer.zero_grad()
        outputs = model(seq1, seq2)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
# Testing on a single batch for demonstration
(sample_seq1, sample_seq2, sample_user1, sample_user2), sample_labels = next(
    iter(keystroke_pair_dataloader)
)
sample_outputs = model(sample_seq1, sample_seq2)

# Print the users and their similarity scores
for u1, u2, score in zip(sample_user1, sample_user2, sample_outputs):
    print(f"User {u1} vs User {u2}: Similarity Score: {score.item():.4f}")
