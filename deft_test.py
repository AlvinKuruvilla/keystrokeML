import matplotlib.pyplot as plt

from core.deft import deft, find_avg_deft_for_deft_distance_and_kit_feature
from core.utils import get_user_by_platform


def show_deft_distance_distribution():
    df = get_user_by_platform(1, 1, 1)
    df2 = get_user_by_platform(2, 1, 1)
    key_list = df["key"]
    key_list2 = df2["key"]
    print(len(key_list))
    input()
    val = []
    val2 = []
    for first, second in zip(key_list, key_list[1:]):
        val.append(deft(first, second))
    for first, second in zip(key_list2, key_list2[1:]):
        val2.append(deft(first, second))
    val = [i for i in val if i != -1]
    val2 = [i for i in val2 if i != -1]
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))
    axs[0].plot(val, marker="o", linestyle="-", color="blue", alpha=0.6)
    axs[0].set_title("User 1")
    axs[0].set_xlabel("Index")
    axs[0].set_ylabel("Value")

    axs[1].plot(val2, marker="s", linestyle="-", color="orange", alpha=0.6)
    axs[1].set_title("User 2")
    axs[1].set_xlabel("Index")
    axs[1].set_ylabel("Value")

    plt.show()


if __name__ == "__main__":
    data1 = []
    data2 = []
    df = get_user_by_platform(1, 1, 2)
    for kit_feature in range(1, 5):
        for deft_val in range(0, 4):
            res = find_avg_deft_for_deft_distance_and_kit_feature(
                df, deft_val, kit_feature
            )
            data1.append(res / (10**-9))
            print(f"KIT{kit_feature} deft {deft_val}: {res/(10**-9)}")
    df2 = get_user_by_platform(1, 1, 1)
    for kit_feature in range(1, 5):
        for deft_val in range(0, 4):
            res = find_avg_deft_for_deft_distance_and_kit_feature(
                df2, deft_val, kit_feature
            )
            data2.append(res / (10**-9))
            print(f"KIT{kit_feature} deft {deft_val}: {res/(10**-9)}")
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))
    axs[0].plot(data1, marker="o", linestyle="-", color="blue", alpha=0.6)
    axs[0].set_title("User 1")
    axs[0].set_xlabel("Index")
    axs[0].set_ylabel("Value")

    axs[1].plot(data2, marker="s", linestyle="-", color="orange", alpha=0.6)
    axs[1].set_title("User 2")
    axs[1].set_xlabel("Index")
    axs[1].set_ylabel("Value")

    plt.show()
