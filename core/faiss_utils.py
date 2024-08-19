import numpy as np
import faiss


def faiss_similarity(user1_feature_list, user2_feature_list):
    vec1 = np.array(user1_feature_list, dtype="float32")
    vec2 = np.array(user2_feature_list, dtype="float32")

    # Check for dimensionality issues
    assert vec1.shape[1] == vec2.shape[1], "Vector dimensions do not match!"

    # Check for NaNs or Infs
    assert not np.isnan(vec1).any(), "vec1 contains NaN values!"
    assert not np.isnan(vec2).any(), "vec2 contains NaN values!"
    assert not np.isinf(vec1).any(), "vec1 contains inf values!"
    assert not np.isinf(vec2).any(), "vec2 contains inf values!"

    # Reshape to fit FAISS input format
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)

    dimension = vec1.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance index

    # Add vectors to the index
    index.add(vec1)

    # Search for the most similar vector in the index (including itself)
    D, I = index.search(vec2, 1)  # Get the distance and index

    similarity_score = 1 / (1 + D[0][0])
    return similarity_score
