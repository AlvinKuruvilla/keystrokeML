from rgb import RGBMatrix, validate_image_pixels
from utils import (
    create_kit_data_from_df,
    map_platform_id_to_initial,
    read_compact_format,
)
from rich.progress import track


for i in track(range(1, 26)):
    if i == 22:
        continue
    for j in range(1, 7):
        for k in range(1, 4):
            print("User ID:", i)
            print("Platform ID:", k)
            print("Session ID:", j)
            df = read_compact_format()
            rem = df[
                (df["user_ids"] == i)
                & (df["session_id"] == j)
                & (df["platform_id"] == k)
            ]
            if rem.empty:
                print(
                    f"Skipping user_id: {i} and platform id: {map_platform_id_to_initial(k)} and session_id: {j}"
                )
                continue
            kit1 = create_kit_data_from_df(rem, 1)
            kit2 = create_kit_data_from_df(rem, 2)
            kit3 = create_kit_data_from_df(rem, 3)
            kit4 = create_kit_data_from_df(rem, 4)
            kit = kit1 | kit2 | kit3 | kit4
            sorted_kit = dict(sorted(kit.items()))
            obj = RGBMatrix(sorted_kit)
            matrix = obj.create_matrix()
            obj.visualize_matrix(
                matrix,
                "result_images/"
                + str(i)
                + "_"
                + map_platform_id_to_initial(k)
                + "_"
                + str(j)
                + ".png",
            )
            # print(
            #     validate_image_pixels(
            #         matrix,
            #         "result_images/"
            #         + str(i)
            #         + "_"
            #         + map_platform_id_to_initial(k)
            #         + "_"
            #         + str(j)
            #         + ".png",
            #     )
            # )
