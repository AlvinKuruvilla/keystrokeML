import numpy as np
from sklearn.metrics import top_k_accuracy_score
from core.deft import flatten_list
from core.faiss_utils import faiss_similarity
from core.feature_heatmap import FeatureHeatMap
from core.feature_table import (
    create_full_user_and_platform_table,
    CKP_SOURCE,
    only_user_id,
    table_to_cleaned_df,
)
from core.utils import all_ids

source = CKP_SOURCE.NORVIG


def test_with_whole_platform():
    rows = create_full_user_and_platform_table(source)
    cleaned = table_to_cleaned_df(rows, source)
    matrix = []

    # Extract features for Facebook and Instagram
    user_ids = all_ids()
    facebook_features = []
    instagram_features = []

    for x in user_ids:
        user_data = only_user_id(cleaned, x)
        num_rows = len(user_data)
        print("Number of platforms are: ", num_rows)

        if (
            num_rows >= 2
        ):  # Ensure there are at least two platforms (Facebook and Instagram)
            # Get Facebook platform features (first row)
            facebook_data = user_data.iloc[0].to_dict()
            facebook_data.pop("user_id", None)
            facebook_data.pop("platform_id", None)
            facebook_feature_vector = flatten_list(list(facebook_data.values()))
            print(facebook_feature_vector)
            input("Feature vector")
            facebook_features.append(facebook_feature_vector)

            # Get Instagram platform features (second row)
            instagram_data = user_data.iloc[0].to_dict()
            instagram_data.pop("user_id", None)
            instagram_data.pop("platform_id", None)
            instagram_feature_vector = flatten_list(list(instagram_data.values()))
            instagram_features.append(instagram_feature_vector)

    # Compute FAISS similarity between Facebook and Instagram features
    for i, fb_vector in enumerate(facebook_features):
        row = []
        for j, ig_vector in enumerate(instagram_features):
            score = faiss_similarity(fb_vector, ig_vector)
            row.append(score)
        matrix.append(row)

    # Plotting the heatmap
    fmap = FeatureHeatMap(source)
    fmap.plot_heatmap(
        matrix,
        "Norvig_FAISS_Facebook_vs_Instagram_similarity_heatmap",
        "Facebook User Index",
        "Instagram User Index",
    )

    return matrix


def top_k_analysis(k_values=[1, 2, 3, 4, 5]):
    # Step 1: Create and clean the data table
    rows = create_full_user_and_platform_table(source)
    cleaned = table_to_cleaned_df(rows, source)

    # Step 2: Prepare data for similarity calculation
    user_ids = all_ids()
    user_vectors = []

    # Populate vectors
    for x in user_ids:
        user1 = only_user_id(cleaned, x).iloc[0].to_dict()
        user1.pop("user_id", None)
        user1.pop("platform_id", None)
        user1_vector = flatten_list(list(user1.values()))
        user_vectors.append((x, user1_vector))

    # Step 3: Calculate similarity scores
    n_users = len(user_vectors)
    similarities = np.zeros((n_users, n_users))

    for i, (user_id1, user1_vector) in enumerate(user_vectors):
        for j, (user_id2, user2_vector) in enumerate(user_vectors):
            if i != j:
                similarities[i, j] = faiss_similarity(user1_vector, user2_vector)
            else:
                similarities[i, j] = 1.0  # Self-similarity should be highest

    # Step 4: Prepare for top-k accuracy calculation
    y_true = np.array(user_ids)
    y_pred = np.argsort(similarities, axis=1)[::-1]

    # Step 5: Calculate top-k accuracy for each k
    top_k_accuracies = {}
    for k in k_values:
        top_k_accuracies[k] = top_k_accuracy_score(y_true, y_pred, k=k)

    return top_k_accuracies


print(test_with_whole_platform())
# top_k_accuracies = top_k_analysis()
# print(top_k_accuracies)
