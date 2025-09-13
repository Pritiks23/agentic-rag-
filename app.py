import os
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from retriever import InMemoryRetriever
from agent import Agent

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY", "demo")
ARIZE_SPACE_KEY = os.getenv("ARIZE_SPACE_KEY", "demo")
LASTMILE_API_KEY = os.getenv("LASTMILE_API_KEY", "demo")

# === Load FLAN-T5 locally ===
print("Loading FLAN-T5 model locally...")
flan_model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(flan_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_name)

def flan_generate(prompt, max_new_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Call stronger HF instruct model for critique & revision ===
def llama_generate(prompt, max_new_tokens=400):
    if not HF_API_TOKEN:
        return "Error: HF_API_TOKEN not set."
    url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens}}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            return str(result)
    except requests.exceptions.RequestException as e:
        return f"Error calling Llama-3: {e}"

# === Agent with self-critique ===
class AgenticRAGAgent(Agent):
    def run(self, query: str):
        docs = self.retriever.retrieve(query, top_k=3)
        context = "\n".join([d['text'] for d in docs])

        # Step 1: Draft answer with FLAN
        draft_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        draft_answer = flan_generate(draft_prompt)

        # Step 2: Critique with Llama-3
        critique_prompt = f"""
You are a critique agent. Analyze the following draft answer.

DRAFT:
{draft_answer}

QUESTION:
{query}

Return ONLY JSON with this schema:
{{
  "missing": ["list of important missing points"],
  "unclear": ["list of unclear statements"],
  "suggestions": ["concrete improvements"]
}}
"""
        critique = llama_generate(critique_prompt)

        # Step 3: Revision with Llama-3
        revision_prompt = f"""
You are a revision agent. Improve the draft answer using the critique.

DRAFT:
{draft_answer}

CRITIQUE:
{critique}

Return ONLY JSON with this schema:
{{
  "final_answer": "Improved answer here",
  "changelog": ["list of changes you made"]
}}
"""
        revision = llama_generate(revision_prompt)

        return {
            "draft": draft_answer,
            "critique": critique,
            "revision": revision
        }

# === Run ===
if __name__ == "__main__":
    query = os.getenv("QUERY", "Summarize the main smart contract vulnerabilities and suggest mitigation strategies.")
    retriever = InMemoryRetriever("data/embeddings.json")
    agent = AgenticRAGAgent(retriever)
    result = agent.run(query)

    print("\n=== INITIAL DRAFT ===\n", result["draft"])
    print("\n=== CRITIQUE ===\n", result["critique"])
    print("\n=== FINAL ANSWER (Revised) ===\n", result["revision"])

    # Log to Arize + Lastmile (same as before)
    from utils import log_to_arize, log_to_lastmile
    log_to_arize(query, result["revision"])
    log_to_lastmile(query, result["revision"])

