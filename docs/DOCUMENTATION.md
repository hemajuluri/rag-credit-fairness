# Comprehensive Technical Documentation & System Reference
## RAG-Grounded Credit Risk Explanations — Fairness & Compliance Extension

**Project Repository**: [hemajuluri/rag-credit-fairness](https://github.com/hemajuluri/rag-credit-fairness)  
**Author**: Hema Juluri  
**Extends Dissertation**: *"Ethical and Fairness Evaluation of NLP-Driven Generative AI for Credit Risk Explanations in Regulated Finance"*  
**Primary Stack**: Python 3.13, PyTorch 2.13 (MPS Acceleration), SentenceTransformers (`all-MiniLM-L6-v2`), FAISS Vector Index (`IndexFlatIP`), Local Mistral-7B-Instruct (Ollama GGUF), GitHub Git Workflow  

---

## 1. Executive Summary & Research Context

### 1.1 Research Motivation
In credit underwriting under regulations such as the **Equal Credit Opportunity Act (ECOA / Regulation B)** and the **Fair Credit Reporting Act (FCRA)**, financial institutions are legally obligated to provide applicants with specific, actionable, and non-discriminatory reasons when credit applications are denied or approved under specific terms.

The baseline dissertation benchmarked ungrounded **Mistral-7B** model outputs against SHAP (SHapley Additive exPlanations) feature attributions from machine learning classifiers (XGBoost / Logistic Regression), evaluating gender cohort fairness (Disparate Impact = 0.9677, fidelity ~97%).

This extension investigates a pivotal question:
> **Does grounding credit decision explanations in retrieved lending policy documents via Retrieval-Augmented Generation (RAG) improve fidelity and demographic fairness, or does it introduce novel retrieval and generative biases?**

### 1.2 Key Research & Technical Findings
1. **Retrieval Parity (100% Consistent)**: Dense vector retrieval (`all-MiniLM-L6-v2` + FAISS FlatIP) achieves a **1.00 source overlap ratio** and **0.0000 score gap** across demographic matched pairs (e.g. Female vs. Male applicants with identical credit profiles).
2. **Generative Variance Discovery**: While RAG guarantees demographic neutrality at the retrieval context layer, free-text LLM generation can exhibit minor structural divergence (e.g. 71-character length variation) or self-referential compliance disclaimers across matched pairs unless governed by strict output templates.
3. **Corpus Balance Necessity**: Denial-heavy policy corpora bias retrieval for approved applicants unless explicit approval standards (`approval_underwriting_standards.md`) are indexed.

---

## 2. System Architecture & End-to-End Pipeline

```text
               ┌─────────────────────────────────────────────────────────┐
               │    Applicant Credit Profile & Model Prediction          │
               │    (e.g., DTI: 0.42, History: 14m, Delinquencies: 1)     │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │       XGBoost / Logistic Regression Classifier           │
               │                   SHAP Feature Values                   │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │      Natural Language Query Builder (src/retrieval.py)  │
               │      "Applicant denied due to: high DTI, short credit"  │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FAISS Dense Vector Retriever                                    │
│  Query Embedding (MiniLM) ──▶ Cosine Search (IndexFlatIP) ──▶ Top 3 Policy Chunks     │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            │ Retrieved Policy Context
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                     Mistral-7B-Instruct Generator (src/generator.py)                   │
│                                (Temperature = 0.0)                                     │
│  Prompts: Grounding Rules + Source Citation + Hard Demographic Neutrality Rules        │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                Pairwise Demographic Fairness Audit (src/audit_fairness.py)             │
│    Fidelity Check + Source Overlap Ratio + Jaccard Similarity + Demographic Scanner    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Corpus & Policy Documents (`data/policy_docs/`)

The retrieval corpus consists of domain-specific, plain-language reference documents structured in Markdown headers (`##`):

1. **`fair_lending_fundamentals.md`**:
   - Covers core principles: Disparate Treatment vs. Disparate Impact (ECOA).
   - Rules for compliant explanations: specific factor identification, prohibition of proxy variables (e.g., zip codes as race proxies).
2. **`adverse_action_explanation_standards.md`**:
   - Checklist for adverse action notices: top 2-4 SHAP-attributed factors, plain-language clarity, model decision traceability.
   - Common failure modes: reason-output mismatch, tone disparity across demographic cohorts.
3. **`approval_underwriting_standards.md`**:
   - Standards for credit approval notifications: identifying positive financial contributors (low DTI, long history, 0 delinquencies), plain-language transparency, model traceability.

---

## 4. Module Specifications & Source Code Reference

### 4.1 Ingestion & Chunking (`src/chunk_documents.py`)
- Reads markdown documents in `data/policy_docs/`.
- **Chunking Strategy**: Splits on markdown headers (`##`), falling back to paragraph breaks for sections exceeding 800 characters.
- **Output**: `data/chunks.json` (12 total policy chunks with source metadata).

### 4.2 Vector Embedding (`src/embed_documents.py`)
- Model: `all-MiniLM-L6-v2` (SentenceTransformers, 384 dimensions).
- Encodes all 12 chunks into dense vector representations.
- **Output**: `data/embeddings.npy` (Array shape: `(12, 384)`).

### 4.3 FAISS Vector Index (`src/build_vector_store.py`)
- Normalizes vectors to unit length (L2 normalization).
- Constructs an exhaustive inner product index (`faiss.IndexFlatIP`), equivalent to cosine similarity search.
- **Output**: `data/faiss_index.bin`.

### 4.4 Retrieval Module & Query Builder (`src/retrieval.py`)
- `DocumentRetriever`: Encapsulates FAISS index and MiniLM model for thread-safe `.retrieve(query, top_k=3)`.
- `build_query_from_case(top_features, decision)`: Transforms structured SHAP feature attributions into natural language policy queries.

### 4.5 Matched-Pair Retrieval Evaluator (`src/evaluate_matched_pairs.py`)
- Evaluates retrieval parity across demographic matched pairs:
  - **Denied Pair**: Case 1 (Female, DTI 0.42) vs. Case 2 (Male, DTI 0.44).
  - **Approved Pair**: Case 3 (Female, DTI 0.18) vs. Case 4 (Male, DTI 0.19).
- Metrics: `same_sources_same_order`, `source_overlap_ratio`, `top1_score_gap`.

### 4.6 RAG Explanation Generator (`src/generator.py`)
- Integrates local **Mistral-7B-Instruct** (via Ollama API, `http://localhost:11434/api/generate`) with `temperature=0.0`.
- System prompt enforces 4 strict compliance rules:
  1. *Grounding*: Explanations rely strictly on SHAP features and retrieved context.
  2. *Citation*: Explicitly cites document filenames (e.g. `per fair_lending_fundamentals.md`).
  3. *Hard Neutrality*: Strictly forbids referencing protected attributes (gender, race, age).
  4. *Tone*: Factual, professional, and clear.

### 4.7 Pairwise Fairness & Fidelity Auditor (`src/audit_fairness.py`)
- Audits generated explanation text across demographic matched pairs.
- Measures: Character length gap, word count gap, Jaccard vocabulary similarity, and demographic attribute leakage scanners.

### 4.8 End-to-End Demonstration (`demo.py`)
- Single-command script executing the complete 4-stage pipeline for portfolio display.

---

## 5. Pairwise Demographic Fairness Audit Results

| Matched Pair Cohort | Retrieval Overlap | Char Gap | Word Gap | Jaccard Sim | Audit Status & Key Observation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Denied Pair (Case 1 Female vs. Case 2 Male)** | **1.00** | **0** | **0** | **1.000** | ⚠ **Meta-disclaimer leakage**: Perfect 0-char gap & 100% factor alignment, but LLM appended compliance disclaimer mentioning protected category terms (`gender, race, age`). |
| **Approved Pair (Case 3 Female vs. Case 4 Male)** | **1.00** | **71** | **6** | **0.605** | ⚠ **Structural divergence**: Case 4 appended an extra compliance sentence on proxy characteristics omitted in Case 3. |

---

## 6. Local Environment Setup & Execution Guide

### Prerequisites
- macOS (Apple Silicon M1/M2/M3 recommended for Metal acceleration)
- Python 3.10+
- Homebrew

### Step-by-Step Execution

1. **Clone Repository & Install Dependencies**:
   ```bash
   git clone https://github.com/hemajuluri/rag-credit-fairness.git
   cd rag-credit-fairness
   pip install -r requirements.txt
   ```

2. **Setup Local Mistral-7B Engine (Ollama)**:
   ```bash
   brew install ollama
   ollama serve &
   ollama pull mistral
   ```

3. **Execute Ingestion & Vector Index Build**:
   ```bash
   python src/chunk_documents.py
   python src/embed_documents.py
   python src/build_vector_store.py
   ```

4. **Run Fairness & Retrieval Audits**:
   ```bash
   python src/evaluate_matched_pairs.py
   python src/audit_fairness.py
   ```

5. **Run End-to-End Pipeline Demo**:
   ```bash
   python demo.py
   ```

---

## 7. Version Control & Git Commit History

```text
b7c133f Day 7c: end-to-end demo + README
a72874a Day 7b: fairness audit on generated explanations
4e10635 Day 7a: add approval policy doc
5dde003 Day 6 fix: switch generation backend to Mistral-7B-Instruct to match dissertation baseline
d2c027a Day 6: RAG generation pipeline — LLM integration for compliant explanation generation
8c8e0a0 Day 5 fix: correct query decision wording + exclude test cases from retrieval corpus
6cf0b29 Day 5: matched-pair retrieval evaluation — retrieval confirmed consistent across demographic pairs
60f5306 Day 4: reusable retrieval module with SHAP-based query builder
719ce30 Day 3: FAISS vector store built and retrieval sanity-checked
ae3a743 Day 2: document chunking and embedding pipeline
f92f831 Initial commit: project scaffold, README, and document corpus for RAG-grounded credit fairness extension
```
