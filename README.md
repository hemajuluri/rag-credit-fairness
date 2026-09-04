# RAG-Grounded Credit Risk Explanations — Fairness Enhancement

This project extends the dissertation *"Ethical and Fairness Evaluation of NLP-Driven Generative AI for Credit Risk Explanations in Regulated Finance"* by grounding credit decision explanations in retrieved lending policy standards, rather than generating explanations purely from an LLM's parametric knowledge.

---

## Why This Matters

The original dissertation benchmarked **Mistral-7B**-generated credit explanations against SHAP feature attributions from XGBoost/Logistic Regression classifiers, auditing for gender fairness (Disparate Impact = 0.9677, fidelity ~97%). 

This extension addresses a critical follow-up question: **Does grounding those explanations in actual lending policy documents via RAG (Retrieval-Augmented Generation) enhance explanation fidelity and demographic fairness?**

---

## Complete Pipeline Architecture

```text
Applicant Features + Credit Classifier Decision (XGBoost/LogReg)
                       │
                       ▼
            SHAP Feature Attribution
                       │
                       ▼
           Retrieval Query Builder ────────── queries ──────────▶ FAISS Vector Store
                       │                                            (Lending Policy Corpus)
                       └───────────── retrieved policy context ◀───────────────┘
                       │
                       ▼
       Local LLM Generator (Mistral-7B-Instruct via Ollama)
       (Grounded in SHAP Values + Retrieved Policy Context)
                       │
                       ▼
  Pairwise Demographic Fairness & Fidelity Audit Report
```

---

## Fairness Audit Findings & Insights

During matched-pair evaluation across demographic cohorts (Case 1 Female vs. Case 2 Male; Case 3 Female vs. Case 4 Male with near-identical risk profiles):

### 1. Retrieval Parity (100% Consistent)
- **Source Overlap Ratio**: `1.00` across all demographic matched pairs.
- **Top-1 Score Gap**: `0.0000`.
- **Finding**: The vector retriever (`all-MiniLM-L6-v2` + FAISS FlatIP) guarantees demographic neutrality at the context retrieval layer when presented with equivalent risk profiles.

### 2. Generative Parity & Variance Analysis
- **Denied Pair (Case 1 vs Case 2)**: 100% factor alignment and 0-character length gap. However, unconstrained generation resulted in self-referential compliance disclaimers citing protected category keywords (`gender, race, age`).
- **Approved Pair (Case 3 vs Case 4)**: 71-character length gap and Jaccard similarity of `0.605` due to subtle structural variation in closing compliance statements between male and female cohorts.
- **Key Takeaway**: RAG ensures retrieval-layer fairness, but free-text LLM generation requires strict output templating or schema enforcement to prevent subtle structural/tonal divergence between demographic cohorts.

---

## Repository Structure

```text
rag-credit-fairness/
├── data/
│   ├── policy_docs/           # Lending policy corpus (Fair Lending, Adverse Action, Approval Standards)
│   ├── case_examples/         # Synthetic applicant cases for evaluation
│   ├── chunks.json            # Parsed document chunks with metadata
│   ├── embeddings.npy         # Dense vector representations (all-MiniLM-L6-v2)
│   └── faiss_index.bin        # FAISS FlatIP vector index
├── src/
│   ├── chunk_documents.py     # Document chunker on markdown headers
│   ├── embed_documents.py     # Sentence-transformers embedding generator
│   ├── build_vector_store.py  # FAISS index builder & similarity search
│   ├── retrieval.py           # Reusable DocumentRetriever class & SHAP query builder
│   ├── evaluate_matched_pairs.py # Demographic pair retrieval consistency evaluator
│   ├── generator.py           # RAG ExplanationGenerator (Mistral-7B-Instruct)
│   └── audit_fairness.py      # End-to-end pairwise fairness & fidelity auditor
├── demo.py                    # Complete end-to-end pipeline demonstration script
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Local Setup & Quick Start

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install & Start Ollama (Mistral-7B)
Install [Ollama](https://ollama.com/) and pull the official `mistral` model:
```bash
brew install ollama
ollama serve &
ollama pull mistral
```

### 3. Run Ingestion & Vector Index Pipeline
```bash
python src/chunk_documents.py
python src/embed_documents.py
python src/build_vector_store.py
```

### 4. Run Matched-Pair Retrieval & Fairness Audits
```bash
python src/evaluate_matched_pairs.py
python src/audit_fairness.py
```

### 5. Run the End-to-End Demo
```bash
python demo.py
```

---

## Relationship to Original Dissertation
- **Original Dissertation Repository**: [github.com/hemajuluri/Ethical-and-fairness](https://github.com/hemajuluri/Ethical-and-fairness)
- This project reuses the original gender-cohort fairness auditing methodology (Disparate Impact, SHAP fidelity scoring) applied to a RAG-grounded architecture.
