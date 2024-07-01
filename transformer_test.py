import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader
from core.transformer_extractor import (
    KeystrokeDataset,
    KeystrokeFeatureExtractorModel,
    extract_features,
    preprocess_data,
)
from core.utils import get_df_by_user_id

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
df = get_df_by_user_id(data, 3)
# Preprocess the data
preprocessed_data = preprocess_data(df)

# Create the dataset and dataloader
dataset = KeystrokeDataset(preprocessed_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Instantiate the model
model = KeystrokeFeatureExtractorModel()


# Extract features from the dataset
transformer_features = extract_features(dataloader, model)
print(transformer_features)
