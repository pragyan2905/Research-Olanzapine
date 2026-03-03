import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings):
        self.index.add(np.array(embeddings))

    def search(self, query_embedding, top_k=5):
        scores, indices = self.index.search(query_embedding, top_k)
        return scores, indices