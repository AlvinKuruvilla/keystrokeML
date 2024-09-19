from core.utils import get_user_by_platform_from_df, read_compact_format


def user_platform_count():
    df = read_compact_format()
    mapping = df.groupby("user_ids")["platform_id"].apply(set).to_dict()
    print(mapping)


df = read_compact_format()
print(get_user_by_platform_from_df(df, 1, 1))
