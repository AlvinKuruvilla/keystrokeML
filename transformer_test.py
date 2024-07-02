import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from core.sample_models.siamese import (
    DataFrameDataset,
    SiameseNetwork,
    compute_similarity,
)
from core.transformer_extractor import get_transformer_features
from core.utils import all_ids, get_df_by_user_id
from rich.progress import track

data = pd.read_csv(
    os.path.join(os.getcwd(), "dataset", "cleaned2.csv"),
    dtype={
        "key": str,
        "press_time": np.float64,
        "release_time": np.float64,
        "platform_id": np.uint8,
        "session_id": np.uint8,
        "user_ids": np.uint8,
    },
)
for id1 in track(all_ids()):
    for id2 in all_ids():
        data1 = get_df_by_user_id(data, id1)
        data2 = get_df_by_user_id(data, id2)

        df1 = get_transformer_features(data1)
        df2 = get_transformer_features(data2)

        dataset1 = DataFrameDataset(df1)
        dataset2 = DataFrameDataset(df2)

        dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=False)
        dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=False)

        # Initialize model
        model = SiameseNetwork(input_dim=128)
        model.eval()  # Set the model to evaluation mode
        # Extract embeddings and compute similarities
        # Extract embeddings and compute the final similarity score
        total_similarity = 0.0
        num_batches = 0

        with torch.no_grad():
            for data1, data2 in zip(dataloader1, dataloader2):
                min_batch_size = min(data1.size(0), data2.size(0))
                data1, data2 = data1[:min_batch_size], data2[:min_batch_size]
                output1, output2 = model(data1, data2)
                similarity = compute_similarity(output1, output2).mean()
                total_similarity += similarity.item()
                num_batches += 1

        # Compute the average similarity score
        final_similarity_score = total_similarity / num_batches
        print(
            "Final Similarity Score (" + str(id1) + "," + str(id2) + "):",
            str(final_similarity_score),
        )
