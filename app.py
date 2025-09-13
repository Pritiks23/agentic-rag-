
#!/usr/bin/env python3
"""
app.py - runs Agentic RAG with self-critiquing agent.
Assumes local FLAN-T5 loaded (transformers + torch). No external HF inference API required.
"""

import os
import argparse
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retriever import InMemoryRetriever
from agent import Agent
from integrations.arize_client import ArizeClient
from integrations.lastmile_client import LastmileClient
from sentence_transformers import SentenceTransformer

# Optional telemetry
ARIZE_API_KEY = os.environ.get("ARIZE_API_KEY")
ARIZE_SPACE_KEY = os.environ.get("ARIZE_SPACE_KEY")
LASTMILE_API_TOKEN = os.environ.get("LASTMILE_API_TOKEN")

# Document source
SOURCES = [
    {
        "id": "frontiers_smart_contracts",
        "url": "https://www.frontiersin.org/journals/blockchain/articles/10.3389/fbloc.2022.814977/full"
    }
]

# Load FLAN-T5 locally once (be mindful of RAM)
print("Loading FLAN-T5 model locally...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
model.eval()
if torch.cuda.is_available():
    model.to("cuda")

# LLM wrapper that uses local model
def query_hf_llm(prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        # Note: we intentionally do NOT pass 'temperature' to avoid the generation flag warning.
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
    except Exception as e:
        return f"Error running local model: {e}"

def fetch_docs():
    docs = []
    for src in SOURCES:
        try:
            r = requests.get(src["url"], timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = "\n".join([p.get_text() for p in paragraphs])
            docs.append({"id": src["id"], "text": text})
        except Exception as e:
            print(f"Error fetching {src['url']}: {e}")
    return docs

def embed_docs(docs):
    model_emb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model_emb.encode([d["text"] for d in docs])
    return embeddings

def main(query):
    print("Fetching docs...")
    docs = fetch_docs()
    if not docs:
        print("No docs found, exiting.")
        return

    print("Embedding docs...")
    embeddings = embed_docs(docs)

    retriever = InMemoryRetriever(docs, embeddings)
    agent = Agent(retriever, query_hf_llm)

    print("Running agentic RAG loop (with self-critique)...")
    results = agent.run(query, top_k=1)

    # Pretty print results
    print("\n=== INITIAL ANSWER ===\n")
    print(results["initial"][:4000])  # truncate long outputs for logs

    print("\n=== CRITIQUE (raw) ===\n")
    print(results["critique"][:4000])

    print("\n=== PARSED CRITIQUE (best-effort) ===\n")
    print(results["critique_parsed"])

    print("\n=== FINAL ANSWER (revised) ===\n")
    print(results["final"][:4000])

    print("\n=== CHANGELOG ===\n")
    for item in results.get("changelog", [])[:10]:
        print("-", item)

    # Arize logging: send the final answer (if keys present)
    if ARIZE_API_KEY and ARIZE_SPACE_KEY:
        arize = ArizeClient(ARIZE_API_KEY, ARIZE_SPACE_KEY)
        arize.log_text(query, results["final"])
    elif ARIZE_API_KEY:
        print("[Arize] Warning: ARIZE_SPACE_KEY missing. Skipping logging.")

    # LastMile logging
    if LASTMILE_API_TOKEN:
        lm = LastmileClient(LASTMILE_API_TOKEN)
        lm.log_evaluation(query, results["final"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()
    main(args.query)
