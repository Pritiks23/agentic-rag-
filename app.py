#!/usr/bin/env python3
"""
Agentic RAG on smart contract vulnerabilities using Hugging Face transformers.
Now running google/flan-t5-large locally instead of via API.
"""

import os
import argparse
import torch
from bs4 import BeautifulSoup
import requests
from retriever import InMemoryRetriever
from agent import Agent
from integrations.arize_client import ArizeClient
from integrations.lastmile_client import LastmileClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ARIZE_API_KEY = os.environ.get("ARIZE_API_KEY")
ARIZE_SPACE_KEY = os.environ.get("ARIZE_SPACE_KEY")
LASTMILE_API_TOKEN = os.environ.get("LASTMILE_API_TOKEN")

SOURCES = [
    {
        "id": "frontiers_smart_contracts",
        "url": "https://www.frontiersin.org/journals/blockchain/articles/10.3389/fbloc.2022.814977/full"
    }
]

# Load model + tokenizer once at startup
print("Loading FLAN-T5 model locally...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

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

def query_hf_llm(prompt):
    """Run prompt through local FLAN-T5 model."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=False
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error running local model: {e}"

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

    print("Running agentic RAG loop...")
    answer = agent.run(query)
    print("\n=== Answer ===\n")
    print(answer)

    if ARIZE_API_KEY and ARIZE_SPACE_KEY:
        arize = ArizeClient(ARIZE_API_KEY, ARIZE_SPACE_KEY)
        arize.log_text(query, answer)
    elif ARIZE_API_KEY:
        print("[Arize] Warning: ARIZE_SPACE_KEY missing. Skipping logging.")

    if LASTMILE_API_TOKEN:
        lm = LastmileClient(LASTMILE_API_TOKEN)
        lm.log_evaluation(query, answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()
    main(args.query)
