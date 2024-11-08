import os
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from rich.progress import track
from core.sample_models.siamese import (
    DataFrameDataset,
    SiameseNetwork,
    compute_similarity,
    train_siamese_network,
)
from core.utils import all_ids, get_df_by_user_id, get_features_dataframe

df = pd.read_csv(
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
        data1 = get_df_by_user_id(df, id1)
        data2 = get_df_by_user_id(df, id2)
        data1_train, data1_test = train_test_split(
            data1, test_size=0.5, random_state=42
        )
        data2_train, data2_test = train_test_split(
            data2, test_size=0.5, random_state=42
        )

        df1_train = get_features_dataframe(data1_train)
        df1_test = get_features_dataframe(data1_test)
        df2_train = get_features_dataframe(data2_train)
        df2_test = get_features_dataframe(data2_test)

        dataset1_train = DataFrameDataset(df1_train)
        dataset2_train = DataFrameDataset(df2_train)
        dataset1_test = DataFrameDataset(df1_test)
        dataset2_test = DataFrameDataset(df2_test)

        dataloader1_train = DataLoader(dataset1_train, batch_size=32, shuffle=True)
        dataloader2_train = DataLoader(dataset2_train, batch_size=32, shuffle=True)
        dataloader1_test = DataLoader(dataset1_test, batch_size=32, shuffle=False)
        dataloader2_test = DataLoader(dataset2_test, batch_size=32, shuffle=False)

        # Initialize model
        model = SiameseNetwork(input_dim=df1_train.shape[1])
        train_siamese_network(model, dataloader1_train, dataloader2_train, id1, id2)
        model.eval()  # Set the model to evaluation mode
        total_similarity = 0.0
        num_batches = 0

        with torch.no_grad():
            for data1, data2 in zip(dataloader1_test, dataloader2_test):
                min_batch_size = min(data1.size(0), data2.size(0))
                data1, data2 = data1[:min_batch_size], data2[:min_batch_size]
                output1, output2 = model(data1, data2)
                similarity = compute_similarity(output1, output2).mean()
                total_similarity += similarity.item()
                num_batches += 1

        final_similarity_score = total_similarity / num_batches
        print(
            "Final Similarity Score (" + str(id1) + "," + str(id2) + "):",
            str(final_similarity_score),
        )
