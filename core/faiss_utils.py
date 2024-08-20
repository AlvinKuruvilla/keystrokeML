import numpy as np
import faiss


def faiss_similarity(user1_feature_list, user2_feature_list):
    vec1 = np.array(user1_feature_list, dtype="float32")
    vec2 = np.array(user2_feature_list, dtype="float32")

    # Ensure dimensions match
    assert vec1.shape == vec2.shape, "Vector dimensions do not match!"

    # Reshape to fit FAISS input format
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    dimension = vec1.shape[1]

    # Create FAISS index with L2 distance
    index = faiss.IndexFlatL2(dimension)

    # Add the first vector to the index
    index.add(vec1)

    # Search for the most similar vector in the index (including itself)
    D, I = index.search(vec2, 1)  # Get the distance and index

    # Convert distance to similarity score
    similarity_score = 1 / (1 + D[0][0])
    return similarity_score
