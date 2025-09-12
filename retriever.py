import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class InMemoryRetriever:
    def __init__(self, docs, embeddings):
        self.docs = docs
        self.embeddings = embeddings

    def retrieve(self, query_embedding, top_k=2):
        sims = cosine_similarity([query_embedding], self.embeddings)[0]
        idxs = np.argsort(sims)[::-1][:top_k]
        return [self.docs[i] for i in idxs]
