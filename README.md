# Agentic RAG for Smart Contract Vulnerabilities

<img width="600" height="215" alt="image" src="https://github.com/user-attachments/assets/3c5933ab-fb0d-4d9f-8170-be715be55d74" />

<img width="512" height="285" alt="image" src="https://github.com/user-attachments/assets/8ccf885a-6e9e-45eb-aec1-e369a9d28f49" />

This project implements an **Agentic Retrieval-Augmented Generation (RAG) system** designed to analyze, summarize, and suggest mitigation strategies for **smart contract vulnerabilities**. Unlike traditional RAG pipelines that only augment responses with retrieved context, **Agentic RAG introduces a reasoning loop** where the agent can iteratively refine its queries to the retriever, evaluate intermediate results, and self-correct before producing a final answer.

The system is fully automated via **GitHub Actions**, requires no local storage, and integrates with **Arize AI** for observability and **Lastmile AI** for evaluation metrics.

---

## 🚀 Key Features

### 1. **Agentic RAG Loop**
- Moves beyond single-pass retrieval → generation.
- Uses an *agent* that can:
  - Re-query the retriever with refined prompts.
  - Inspect intermediate evidence.
  - Compose more accurate and robust final outputs.
- Impact: Improves resilience against **hallucinations** and **context omission**, which is critical when dealing with blockchain security data.

### 2. **Local LLM Inference with FLAN-T5**
- Runs `google/flan-t5-large` (or `flan-t5-base` for lighter compute) locally using Hugging Face `transformers`.
- Avoids reliance on external inference APIs (e.g., OpenAI/HF Inference endpoints).
- Deterministic, auditable, and cost-free once downloaded.

### 3. **Smart Contract Vulnerability Corpus**
- Ingests real-world references such as:
  - [Frontiers in Blockchain research](https://www.frontiersin.org/journals/blockchain/articles/10.3389/fbloc.2022.814977/full)
  - [ArXiv papers on smart contract analysis](https://arxiv.org/pdf/2212.05099)
- Documents are embedded with `sentence-transformers/all-MiniLM-L6-v2` for efficient semantic retrieval.

### 4. **Observability & Evaluation**
- **Arize AI**: Logs queries and generated answers, enabling monitoring of output quality over time.
- **Lastmile AI**: Provides evaluation hooks for grounding, factual correctness, and performance metrics.

### 5. **CI/CD via GitHub Actions**
- End-to-end pipeline runs in the cloud.
- No local environment setup or disk storage needed.
- Reproducible runs with environment variables stored as **GitHub Secrets**.

---

## 📂 Project Structure

