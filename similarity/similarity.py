from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimilarityEngine:
    def __init__(self, k=5):
        self.k = k

    def find_similar(self, new_vector, existing_vectors):
        similarities = cosine_similarity(new_vector, existing_vectors)[0]
        top_k_idx = np.argsort(similarities)[-self.k:][::-1]
        return top_k_idx, similarities[top_k_idx]
