from tqdm import tqdm
import re
from core.bigrams import oxford_bigrams
from core.feature_table import alpha_word_bigrams, most_common_kepairs
from core.utils import (
    all_ids,
    create_kit_data_from_df,
    get_user_by_platform,
    map_platform_id_to_initial,
)


def clean_strings(strings):
    cleaned_strings = []

    for s in strings:
        # Remove extraneous single quotes and retain the actual content
        cleaned = re.sub(r"'\s*|\s*'", "", s)

        # Remove any extra spaces
        cleaned = cleaned.strip()

        # Append the cleaned string to the result list
        cleaned_strings.append(cleaned)

    return cleaned_strings


def raw_ckp_test():
    ckps = most_common_kepairs()
    for i in tqdm(all_ids()):
        for j in range(1, 4):
            for k in range(1, 7):
                df = get_user_by_platform(i, j, k)
                if df.empty:
                    print(
                        f"Skipping User: {i}, platform: {map_platform_id_to_initial(j)}, session: {k}"
                    )
                    continue
                for x in range(1, 5):
                    test = list(
                        create_kit_data_from_df(df, x, use_seperator=False).keys()
                    )
                    for ckp in ckps:
                        if ckp in test:
                            print(f"Found ckp: {ckp} in keyset")
                        else:
                            print(
                                f"Did not find ckp: {ckp} in keyset User: {i}, platform: {map_platform_id_to_initial(j)}, session: {k}"
                            )


def alpha_bigrams_test():
    ckps = alpha_word_bigrams()
    for i in tqdm(all_ids()):
        for j in range(1, 4):
            for k in range(1, 7):
                df = get_user_by_platform(i, j, k)
                if df.empty:
                    print(
                        f"Skipping User: {i}, platform: {map_platform_id_to_initial(j)}, session: {k}"
                    )
                    continue
                for x in range(1, 5):
                    test = list(
                        create_kit_data_from_df(df, x, use_seperator=False).keys()
                    )
                    for ckp in ckps:
                        if ckp in test:
                            print(f"Found ckp: {ckp} in keyset")
                        else:
                            print(f"Did not find ckp: {ckp} in keyset")


def emory_test():
    ckps = alpha_word_bigrams()
    for i in tqdm(all_ids()):
        for j in range(1, 4):
            df = get_user_by_platform(i, j)
            if df.empty:
                print(f"Skipping User: {i}, platform: {map_platform_id_to_initial(j)}")
                continue
            for x in range(1, 5):
                test = list(create_kit_data_from_df(df, x, use_seperator=False).keys())
                print(test)
                input()
                for ckp in ckps:
                    if ckp in test:
                        print(f"Found ckp: {ckp} in keyset")
                    else:
                        print(f"Did not find ckp: {ckp} in keyset")


def basic():
    for i in tqdm(all_ids()):
        for j in range(1, 4):
            df = get_user_by_platform(i, j)
            if df.empty:
                print(f"Skipping User: {i}, platform: {map_platform_id_to_initial(j)}")
                continue
            for x in range(1, 5):
                test = clean_strings(
                    list(create_kit_data_from_df(df, x, use_seperator=False).keys())
                )
                print(test)
                print("in" in test)
        input()


basic()
