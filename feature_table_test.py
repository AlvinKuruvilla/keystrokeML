import pandas as pd
from tqdm import tqdm
from core.feature_table import (
    KeystrokeFeatureTable,
    fill_empty_row_values,
    get_ckps,
    most_common_kepairs,
    only_platform_id,
    flatten_kit_feature_columns,
    CKP_SOURCE,
    drop_empty_list_columns,
)
from core.utils import all_ids, get_user_by_platform, map_platform_id_to_initial


def is_empty_list(x):
    return isinstance(x, list) and len(x) == 0


rows = []
source = CKP_SOURCE.ALPHA_WORDS
# for i in tqdm(all_ids()):
#     for j in range(1, 4):
#         for k in range(1, 7):
#             df = get_user_by_platform(i, j, k)
#             if df.empty:
#                 print(
#                     f"Skipping User: {i}, platform: {map_platform_id_to_initial(j)}, session: {k}"
#                 )
#                 continue
#             print(f"User: {i}, platform: {map_platform_id_to_initial(j)}, session: {k}")
#             table = KeystrokeFeatureTable()
#             table.find_kit_from_most_common_keypairs(df, CKP_SOURCE.OXFORD_EMORY)
#             table.find_deft_for_df(df=df)
#             table.add_user_platform_session_identifiers(i, j, k)

#             row = table.as_df()
#             rows.append(row)

for i in tqdm(all_ids()):
    for j in range(1, 4):
        df = get_user_by_platform(i, j)
        if df.empty:
            print(f"Skipping User: {i}, platform: {map_platform_id_to_initial(j)}")
            continue
        print(f"User: {i}, platform: {map_platform_id_to_initial(j)}")
        table = KeystrokeFeatureTable()
        table.find_kit_from_most_common_keypairs(df, source)
        # table.find_deft_for_df(df=df)
        table.add_user_platform_session_identifiers(i, j, None)

        row = table.as_df()
        rows.append(row)

combined_df = pd.concat(rows, axis=0)
print(combined_df.columns)
empty_list_count = combined_df.stack().map(is_empty_list).sum()
print(f"Number of cells containing empty lists: {empty_list_count}")
# print(list(only_platform_id(combined_df, 1)))
full_df = fill_empty_row_values(combined_df, get_ckps(source))
empty_list_count = full_df.stack().map(is_empty_list).sum()
print(f"Number of cells containing empty lists (post fill): {empty_list_count}")
fixed_df = flatten_kit_feature_columns(full_df, get_ckps(source))
print(drop_empty_list_columns(fixed_df))
