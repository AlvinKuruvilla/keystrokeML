import torch
import torch.nn as nn
from torch.utils.data import Dataset


# Custom Dataset to handle the dataframes
class DataFrameDataset(Dataset):
    def __init__(self, df):
        self.df = df.values.astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.df[idx])


# Simple Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)

    def forward_one(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


# Function to compute cosine similarity
def compute_similarity(embedding1, embedding2):
    return nn.functional.cosine_similarity(embedding1, embedding2)


# Training loop for the Siamese Network
def train_siamese_network(
    model, dataloader1, dataloader2, id1, id2, num_epochs=10, learning_rate=0.001
):
    criterion = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        for data1, data2 in zip(dataloader1, dataloader2):
            min_batch_size = min(data1.size(0), data2.size(0))
            data1, data2 = data1[:min_batch_size], data2[:min_batch_size]
            output1, output2 = model(data1, data2)

            # Create labels: +1 if same, -1 if different
            labels = (
                torch.ones(min_batch_size)
                if id1 == id2
                else -torch.ones(min_batch_size)
            )

            loss = criterion(output1, output2, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Debugging: Check gradients
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Grad {name}: {param.grad.abs().mean()}")

        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
