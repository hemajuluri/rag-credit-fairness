"""
demo.py

RAG Credit Fairness Extension — End-to-End Pipeline Demo

Demonstrates the complete RAG credit-decision explanation workflow:
1. Input: Applicant SHAP feature attributions & model decision
2. Query Builder: Generates natural language policy query
3. Vector Retriever: FAISS index retrieves top matching lending policy chunks
4. Generator: Local Mistral-7B-Instruct generates grounded, compliant explanation notice

Usage:
    python demo.py
"""

import sys
from pathlib import Path

# Ensure src/ is on python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from generator import ExplanationGenerator


def run_demo():
    print("==================================================================")
    print("  RAG-GROUNDED CREDIT FAIRNESS PIPELINE — END-TO-END DEMO")
    print("==================================================================")
    print("Model Backend: Mistral-7B-Instruct (Local Ollama GGUF)")
    print("Embedding Model: all-MiniLM-L6-v2 | Vector Store: FAISS FlatIP\n")

    # Sample applicant profile (Case 1: Denied)
    sample_applicant = {
        "id": "Demo Case (Applicant #1042)",
        "gender": "F",
        "decision": "Denied",
        "features": [
            {"feature": "debt-to-income ratio", "value": "0.42", "direction": "high"},
            {"feature": "credit history length", "value": "14 months", "direction": "short"},
            {"feature": "delinquent accounts in past 12 months", "value": "1", "direction": "one"},
        ],
    }

    print("--- [STAGE 1: INPUT SHAP FEATURE ATTRIBUTIONS] ---")
    print(f"Applicant ID: {sample_applicant['id']}")
    print(f"Model Decision: {sample_applicant['decision']}")
    print("Top Decision Factors:")
    for f in sample_applicant["features"]:
        print(f"  - {f['direction']} {f['feature']} (value: {f['value']})")

    generator = ExplanationGenerator()
    result = generator.generate_explanation(sample_applicant, top_k=3)

    print("\n--- [STAGE 2: RETRIEVAL QUERY BUILDER] ---")
    print(f"Generated Policy Query: '{result['query']}'")

    print("\n--- [STAGE 3: FAISS VECTOR STORE RETRIEVAL] ---")
    print(f"Retrieved Top {len(result['retrieved_chunks'])} Policy Chunks:")
    for i, c in enumerate(result["retrieved_chunks"], 1):
        print(f"  {i}. [{c['score']:.3f}] {c['source_file']}: {c['text'][:80]}...")

    print("\n--- [STAGE 4: MISTRAL-7B RAG EXPLANATION GENERATION] ---")
    print(result["explanation"])

    print("\n==================================================================")
    print("  DEMO COMPLETE — Explanation grounded in policy & SHAP values")
    print("==================================================================\n")


if __name__ == "__main__":
    run_demo()
