import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, value, key, query):
        attention, _ = self.attention(query, key, value)
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


# Define the Bi-Encoder model
class BiEncoder(nn.Module):
    def __init__(self, embed_size, heads, depth):
        super(BiEncoder, self).__init__()
        self.embed_size = embed_size
        self.position_embedding = nn.Embedding(1000, embed_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads) for _ in range(depth)]
        )
        self.linear_projection = nn.Linear(embed_size, embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size), nn.ReLU(), nn.Linear(embed_size, 1)
        )

    def forward(self, x):
        N, seq_length = x.shape
        positions = (
            torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length).to(x.device)
        )
        out = self.position_embedding(positions)
        for layer in self.layers:
            out = layer(out, out, out)
        out = self.linear_projection(out)
        out = out.mean(dim=1)
        out = self.feed_forward(out)
        return out


class KeystrokeDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.data.iloc[:, :-1].values)
        self.labels = self.data.iloc[:, -1].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.float32
        )
