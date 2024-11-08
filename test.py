from core.utils import all_ids_with_full_platforms, read_compact_format


def user_platform_count():
    df = read_compact_format()
    mapping = df.groupby("user_ids")["platform_id"].apply(set).to_dict()
    print(mapping)


print(all_ids_with_full_platforms())
