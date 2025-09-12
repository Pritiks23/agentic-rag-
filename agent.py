from sentence_transformers import SentenceTransformer

class Agent:
    def __init__(self, retriever, llm_call):
        self.retriever = retriever
        self.llm_call = llm_call
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def run(self, query):
        query_emb = self.embed_model.encode([query])[0]
        docs = self.retriever.retrieve(query_emb, top_k=1)
        context = "\n---\n".join([d["text"] for d in docs])
        prompt = f"Context:\n{context}\n\nQuery:\n{query}"
        return self.llm_call(prompt)
