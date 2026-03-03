from sentence_transformers import CrossEncoder


class ReRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents):
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        return scores