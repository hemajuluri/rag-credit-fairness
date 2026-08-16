# RAG-Grounded Credit Risk Explanations — Fairness Enhancement

This project extends the dissertation *"Ethical and Fairness Evaluation of NLP-Driven Generative AI for Credit Risk Explanations in Regulated Finance"* by grounding the LLM's credit-decision explanations in retrieved policy documents and prior case precedents, rather than generating explanations purely from the model's parametric knowledge.

## Why this matters

The original dissertation benchmarked Mistral-7B-generated credit explanations against SHAP feature attributions from XGBoost/Logistic Regression classifiers, auditing for gender fairness (Disparate Impact = 0.9677, fidelity ~97%). This project asks a follow-up question: **does grounding those explanations in actual lending policy documents and precedent cases change their fidelity or fairness — for better or worse?**

Retrieval-augmented generation (RAG) is increasingly how regulated-industry LLM applications are built in practice, since ungrounded generation is a compliance risk. This project tests whether RAG grounding is also a *fairness* improvement, not just a factuality one.

## Project status

🚧 In progress — Week 1 of 6. See `docs/progress.md` (added as the project develops) for session-by-session updates.

## Architecture (planned)

```
Applicant features + model decision (XGBoost/LogReg)
        │
        ▼
   SHAP feature attribution
        │
        ▼
   Retrieval module ──── queries ────▶ Vector store (policy docs + case precedents)
        │                                        │
        └──────────────── retrieved context ◀────┘
        │
        ▼
   LLM (Mistral-7B) generates explanation
   grounded in SHAP values + retrieved context
        │
        ▼
   Fairness re-audit (gender cohorts) vs. dissertation baseline
```

## Repo structure

```
rag-credit-fairness/
├── data/
│   ├── policy_docs/       # lending policy & fair-lending reference documents (the retrieval corpus)
│   └── case_examples/     # synthetic applicant cases used to test retrieval + generation
├── src/                   # embedding, retrieval, and generation modules (added Week 2+)
├── notebooks/             # exploratory analysis, fairness re-audit (added Week 5)
└── README.md
```

## Milestones

| Week | Milestone |
|---|---|
| 1 | Scaffold + document corpus (this session) |
| 2 | Embedding + vector store (FAISS/Chroma) |
| 3 | Retrieval module |
| 4 | Generation module (SHAP + retrieved context → explanation) |
| 5 | Fairness re-audit vs. dissertation baseline |
| 6 | Evaluation write-up |

## Setup

```bash
pip install -r requirements.txt
```

(`requirements.txt` added once the embedding/retrieval stack is chosen in Week 2.)

## Relationship to the original dissertation

Dissertation repo: [github.com/hemajuluri/Ethical-and-fairness](https://github.com/hemajuluri/Ethical-and-fairness). This project is a standalone extension, not a fork — it reuses the fairness auditing *methodology* (gender cohort Disparate Impact, fidelity scoring) but applies it to a new RAG-based generation pipeline.
