from feature_matrix import (
    combine_into_feature_matrix,
    create_kht_feature_matrix,
    create_kit_feature_matrix,
)
from utils import create_kht_data_from_df, create_kit_data_from_df, read_compact_format
from rich.progress import track

for i in track(range(1, 27)):
    if i == 22:
        continue
    print("User ID:", i)
    df = read_compact_format()
    rem = df[(df["user_ids"] == i)]
    kht_data = create_kht_data_from_df(rem)
    kit_data = create_kit_data_from_df(rem, 1, False)
    kht_feature_matrix = create_kht_feature_matrix(kht_data)
    kit_feature_matrix = create_kit_feature_matrix(kit_data)
    print(combine_into_feature_matrix(kht_feature_matrix, kit_feature_matrix))
