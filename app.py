#!/usr/bin/env python3
"""
Agentic RAG on smart contract vulnerabilities using Hugging Face free LLM.
Updated to use facebook/bart-large-cnn model.
"""

import os
import requests
import argparse
from bs4 import BeautifulSoup
from retriever import InMemoryRetriever
from agent import Agent
from integrations.arize_client import ArizeClient
from integrations.lastmile_client import LastmileClient
from sentence_transformers import SentenceTransformer

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
ARIZE_API_KEY = os.environ.get("ARIZE_API_KEY")
ARIZE_SPACE_KEY = os.environ.get("ARIZE_SPACE_KEY")
LASTMILE_API_TOKEN = os.environ.get("LASTMILE_API_TOKEN")

# Easy-to-fetch article URL
SOURCES = [
    {
        "id": "frontiers_smart_contracts",
        "url": "https://www.frontiersin.org/journals/blockchain/articles/10.3389/fbloc.2022.814977/full"
    }
]

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
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode([d["text"] for d in docs])
    return embeddings

def query_hf_llm(prompt):
    if not HF_API_TOKEN:
        return "Error: HF_API_TOKEN not set."
    
    url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300}}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()[0]["generated_text"]
    except requests.exceptions.RequestException as e:
        return f"Error calling HF LLM: {e}"

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

    # Arize logging
    if ARIZE_API_KEY and ARIZE_SPACE_KEY:
        arize = ArizeClient(ARIZE_API_KEY, ARIZE_SPACE_KEY)
        arize.log_text(query, answer)
    elif ARIZE_API_KEY:
        print("[Arize] Warning: ARIZE_SPACE_KEY missing. Skipping logging.")

    # LastMile logging
    if LASTMILE_API_TOKEN:
        lm = LastmileClient(LASTMILE_API_TOKEN)
        lm.log_evaluation(query, answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()
    main(args.query)

