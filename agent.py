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
        self.retriever = retriever
        self.llm_call = llm_call
        self.embed_model = embed_model or SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def run(self, query, top_k=1):
        query_emb = self.embed_model.encode([query])[0]
        docs = self.retriever.retrieve(query_emb, top_k=top_k)
        context = "\n---\n".join([d["text"] for d in docs])

        # Initial answer
        initial_prompt = (
            "You are an expert in smart contract security. Use the context below to answer the user's query.\n\n"
            f"Context:\n{context}\n\n"
            f"User Query:\n{query}\n\n"
            "Instruction: produce a concise, structured answer with concrete mitigation steps. "
            "When you cite specifics, prefer phrasing like 'According to the context ...'."
        )
        initial = self.llm_call(initial_prompt)

        # Critique step
        critique_prompt = (
            "You are an expert reviewer. Given the Context, Query, and the Initial Answer, "
            "produce a structured critique. Output valid JSON with the following keys:\n"
            "  - errors: list of factual errors (short strings)\n"
            "  - missing: list of missing important points that should be added\n"
            "  - unclear: list of unclear statements to rewrite\n"
            "  - suggestions: list of concrete suggestions to improve the answer\n\n"
            "If there are no items for a key, return an empty list for that key.\n"
            "Important: Return only valid JSON. Do not include commentary or explanation.\n\n"
            "Example Output:\n"
            "{\n"
            '  "errors": [],\n'
            '  "missing": ["No mention of reentrancy attacks", "No mitigation strategies provided"],\n'
            '  "unclear": ["\'systematic review\' is vague"],\n'
            '  "suggestions": ["Add examples of vulnerabilities", "Include concrete mitigation steps"]\n'
            "}\n\n"
            f"Context:\n{context}\n\n"
            f"Query:\n{query}\n\n"
            f"Initial Answer:\n{initial}\n\n"
            "Return only the JSON object."
        )
        critique = self.llm_call(critique_prompt)

        # Parse critique
        critique_parsed = None
        try:
            critique_parsed = json.loads(critique)
        except Exception:
            m = re.search(r'\{.*?\}', critique, re.S)
            if m:
                try:
                    critique_parsed = json.loads(m.group(0))
                except Exception:
                    critique_parsed = None

        # Revision step
        revision_instructions = (
            "Revise the Initial Answer so it incorporates the critique. "
            "Be explicit: return a JSON with keys 'final_answer' (string) and 'changelog' (list of short bullets describing what you changed). "
            "Keep final_answer concise and actionable."
        )

        if critique_parsed:
            revision_prompt = (
                f"Context:\n{context}\n\n"
                f"Query:\n{query}\n\n"
                f"Initial Answer:\n{initial}\n\n"
                f"Critique:\n"
                f"- Errors: {critique_parsed.get('errors', [])}\n"
                f"- Missing: {critique_parsed.get('missing', [])}\n"
                f"- Unclear: {critique_parsed.get('unclear', [])}\n"
                f"- Suggestions: {critique_parsed.get('suggestions', [])}\n\n"
                f"{revision_instructions}"
            )
        else:
            revision_prompt = (
                f"Context:\n{context}\n\n"
                f"Query:\n{query}\n\n"
                f"Initial Answer:\n{initial}\n\n"
                f"The following critique may be unstructured. Try to extract actionable suggestions:\n{critique}\n\n"
                f"{revision_instructions}"
            )

        revision = self.llm_call(revision_prompt)

        # Parse revision
        final_answer = revision
        changelog = None
        try:
            parsed_rev = json.loads(revision)
            if isinstance(parsed_rev, dict):
                final_answer = parsed_rev.get("final_answer", final_answer)
                changelog = parsed_rev.get("changelog", None)
        except Exception:
            m2 = re.search(r'\{.*?\}', revision, re.S)
            if m2:
                try:
                    parsed_rev = json.loads(m2.group(0))
                    final_answer = parsed_rev.get("final_answer", final_answer)
                    changelog = parsed_rev.get("changelog", None)
                except Exception:
                    changelog = None

        # Fallback changelog
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

