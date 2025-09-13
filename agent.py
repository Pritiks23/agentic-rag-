# agent.py
"""
Agent with self-critiquing loop.

Behavior:
 - retrieve context
 - generate initial answer
 - ask the model to critique that answer (structured JSON if possible)
 - generate a revised final answer that incorporates the critique
 - return all three pieces (initial, critique, final)
"""

import json
import re
from sentence_transformers import SentenceTransformer

class Agent:
    def __init__(self, retriever, llm_call, embed_model=None):
        """
        retriever: object with retrieve(query_embedding, top_k) -> list[{"id","text"}]
        llm_call: function(prompt:str) -> str  (should call your local HF model)
        embed_model: SentenceTransformer instance (optional)
        """
        self.retriever = retriever
        self.llm_call = llm_call
        if embed_model is None:
            self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            self.embed_model = embed_model

    def run(self, query, top_k=1):
        """
        Run a single self-critiquing RAG iteration.
        Returns a dict: { initial, critique, critique_parsed, final }
        """
        # 1) embed + retrieve
        query_emb = self.embed_model.encode([query])[0]
        docs = self.retriever.retrieve(query_emb, top_k=top_k)
        context = "\n---\n".join([d["text"] for d in docs])

        # 2) initial answer
        initial_prompt = (
            "You are an expert in smart contract security. Use the context below to answer the user's query.\n\n"
            f"Context:\n{context}\n\n"
            f"User Query:\n{query}\n\n"
            "Instruction: produce a concise, structured answer with concrete mitigation steps. "
            "When you cite specifics, prefer phrasing like 'According to the context ...'."
        )
        initial = self.llm_call(initial_prompt)

        # 3) critique step - ask model to produce JSON-like critique
        critique_prompt = (
            "You are an expert reviewer. Given the Context, Query, and the Initial Answer, "
            "produce a structured critique. Output valid JSON with the following keys:\n"
            "  - errors: list of factual errors (short strings)\n"
            "  - missing: list of missing important points that should be added\n"
            "  - unclear: list of unclear statements to rewrite\n"
            "  - suggestions: list of concrete suggestions to improve the answer\n\n"
            "If there are no items for a key, return an empty list for that key.\n\n"
            f"Context:\n{context}\n\n"
            f"Query:\n{query}\n\n"
            f"Initial Answer:\n{initial}\n\n"
            "Return only the JSON object (no extra commentary)."
        )
        critique = self.llm_call(critique_prompt)

        # 4) try to parse critique into JSON (best-effort)
        critique_parsed = None
        try:
            critique_parsed = json.loads(critique)
        except Exception:
            # try to extract a JSON substring
            m = re.search(r'\{.*\}', critique, re.S)
            if m:
                try:
                    critique_parsed = json.loads(m.group(0))
                except Exception:
                    critique_parsed = None

        # 5) revision step - incorporate critique
        revision_instructions = (
            "Revise the Initial Answer so it incorporates the critique. "
            "Be explicit: return a JSON with keys 'final_answer' (string) and 'changelog' (list of short bullets describing what you changed). "
            "Keep final_answer concise and actionable."
        )
        revision_prompt = (
            f"Context:\n{context}\n\n"
            f"Query:\n{query}\n\n"
            f"Initial Answer:\n{initial}\n\n"
            f"Critique (raw):\n{critique}\n\n"
            f"{revision_instructions}"
        )
        revision = self.llm_call(revision_prompt)

        # Try to parse revision JSON (best-effort)
        final_answer = revision
        changelog = None
        try:
            parsed_rev = json.loads(revision)
            if isinstance(parsed_rev, dict):
                final_answer = parsed_rev.get("final_answer", final_answer)
                changelog = parsed_rev.get("changelog", None)
        except Exception:
            # extract JSON if present
            m2 = re.search(r'\{.*\}', revision, re.S)
            if m2:
                try:
                    parsed_rev = json.loads(m2.group(0))
                    final_answer = parsed_rev.get("final_answer", final_answer)
                    changelog = parsed_rev.get("changelog", None)
                except Exception:
                    changelog = None

        # If changelog not provided, create a tiny one from critique_parsed suggestions
        if changelog is None:
            if critique_parsed and isinstance(critique_parsed.get("suggestions", []), list):
                changelog = [s for s in critique_parsed.get("suggestions", [])][:6]
            else:
                changelog = []

        return {
            "initial": initial,
            "critique": critique,
            "critique_parsed": critique_parsed,
            "final": final_answer,
            "changelog": changelog
        }

