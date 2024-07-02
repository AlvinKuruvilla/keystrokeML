import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer


# Define a custom dataset class
class KeystrokeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


# Define the Transformer model for feature extraction
class KeystrokeFeatureExtractorModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name="bert-base-uncased",
        hidden_size=768,
        num_features=128,
        dropout_rate=0.3,
    ):
        super(KeystrokeFeatureExtractorModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_features)
        self.batch_norm2 = nn.BatchNorm1d(num_features)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take the CLS token output
        x = self.dropout(cls_output)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        features = torch.relu(x)
        return features


# Preprocess the data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def preprocess_data(data):
    texts = data["key"].apply(
        lambda x: " ".join(x)
    )  # Convert keystroke sequences to space-separated strings
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True)

    # Use .assign to avoid SettingWithCopyWarning
    data = data.assign(
        input_ids=encodings["input_ids"], attention_mask=encodings["attention_mask"]
    )

    return data


# Function to extract features
def extract_features(dataloader, model):
    model.eval()
    features_list = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            features = model(input_ids=input_ids, attention_mask=attention_mask)
            features_list.append(features.cpu().numpy())
    features = np.concatenate(features_list, axis=0)

    # Convert to DataFrame
    num_features = features.shape[1]
    columns = [f"tf{i}" for i in range(num_features)]
    return pd.DataFrame(features, columns=columns)


def get_transformer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Preprocess the data
    preprocessed_data = preprocess_data(df)

    # Create the dataset and dataloader
    dataset = KeystrokeDataset(preprocessed_data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Instantiate the model
    model = KeystrokeFeatureExtractorModel()

    # Extract features from the dataset
    return extract_features(dataloader, model)
