from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer


def cluster_papers(abstracts, num_clusters=2):
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = model.encode(abstracts, normalize_embeddings=True)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    return labels