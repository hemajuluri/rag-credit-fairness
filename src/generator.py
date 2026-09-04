"""
generator.py

Day 6: RAG Generation Pipeline (Mistral-7B-Instruct Backend)

Wires the DocumentRetriever (Days 4-5) into a local Mistral-7B-Instruct call
(via Ollama) to generate RAG-grounded, compliant credit decision explanations
based on applicant SHAP feature attributions and retrieved policy documents.

Model choice: Mistral-7B-Instruct (matching dissertation baseline & README.md).

Compliance & Fairness Rules Enforced:
1. Grounding: Explanation relies strictly on SHAP features and retrieved context.
2. Citation: Policy sources are cited explicitly (e.g. per fair_lending_fundamentals.md).
3. Hard Neutrality: Protected attributes (gender, race, age) are NEVER referenced.
4. Determinism: Low temperature (0.0) is used for factual compliance text.
"""

import json
import os
import urllib.request
from pathlib import Path

from retrieval import DocumentRetriever, build_query_from_case

MODEL_NAME = "mistral"
OLLAMA_URL = "http://localhost:11434/api/generate"


class ExplanationGenerator:
    """Combines retrieval and local Mistral-7B generation to produce grounded credit explanations."""

    def __init__(
        self,
        retriever: DocumentRetriever = None,
        model_name: str = MODEL_NAME,
        temperature: float = 0.0,
        ollama_url: str = OLLAMA_URL,
    ):
        self.retriever = retriever or DocumentRetriever()
        self.model_name = model_name
        self.temperature = temperature
        self.ollama_url = ollama_url

    def _build_prompt(self, case: dict, retrieved_chunks: list[dict]) -> str:
        decision = case.get("decision", "Denied")
        features = case.get("features", [])

        feature_str = "\n".join(
            [f"- {f.get('direction', '')} {f['feature']} (value: {f.get('value', 'N/A')})" for f in features]
        )

        context_str = "\n\n".join(
            [f"--- Document: {c['source_file']} (Relevance Score: {c['score']:.3f}) ---\n{c['text']}" for c in retrieved_chunks]
        )

        prompt = f"""You are a compliance-focused credit risk analyst generating official credit decision explanations.

### Applicant Decision & Factors
- Official Decision: {decision}
- Key Decision Factors (SHAP feature attributions):
{feature_str}

### Retrieved Policy Context Documents
{context_str}

### Strict Compliance Rules:
1. Grounding: State the decision and explain it relying ONLY on the applicant's key decision factors and the retrieved policy context above. Do NOT invent, assume, or reference any external factors or rules.
2. Source Citation: You MUST explicitly cite the relevant source document filename(s) (e.g., "per fair_lending_fundamentals.md" or "according to adverse_action_explanation_standards.md") when referencing policy rules or explanation standards.
3. Hard Neutrality Rule: NEVER reference, mention, or imply gender, race, age, ethnicity, national origin, or any other protected demographic attribute under any circumstances.
4. Tone: Factual, professional, and clear.

Generate the official credit decision explanation below:"""
        return prompt

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        req = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data.get("response", "").strip()
        except Exception as e:
            raise RuntimeError(
                f"Failed to communicate with Ollama service at {self.ollama_url}. "
                f"Make sure 'ollama serve' is running. Error: {e}"
            )

    def generate_explanation(self, case: dict, top_k: int = 3) -> dict:
        decision = case.get("decision", "Denied")
        features = case.get("features", [])

        query = build_query_from_case(features, decision=decision)
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)

        prompt = self._build_prompt(case, retrieved_chunks)

        explanation_text = self._call_ollama(prompt)

        return {
            "case_id": case.get("id", "Unknown"),
            "gender": case.get("gender", "N/A"),
            "decision": decision,
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "prompt": prompt,
            "explanation": explanation_text,
        }


if __name__ == "__main__":
    generator = ExplanationGenerator()

    # Test Case 1 (Denied)
    test_case_denied = {
        "id": "Case 1",
        "gender": "F",
        "decision": "Denied",
        "features": [
            {"feature": "debt-to-income ratio", "value": "0.42", "direction": "high"},
            {"feature": "credit history length", "value": "14 months", "direction": "short"},
            {"feature": "delinquent accounts in past 12 months", "value": "1", "direction": "one"},
        ],
    }

    # Test Case 3 (Approved)
    test_case_approved = {
        "id": "Case 3",
        "gender": "F",
        "decision": "Approved",
        "features": [
            {"feature": "debt-to-income ratio", "value": "0.18", "direction": "low"},
            {"feature": "credit history length", "value": "86 months", "direction": "long"},
            {"feature": "delinquent accounts", "value": "0", "direction": "no"},
        ],
    }

    print("==================================================================")
    print("DAY 6: END-TO-END RAG GENERATION TEST (MISTRAL-7B-INSTRUCT)")
    print("==================================================================\n")

    for case in [test_case_denied, test_case_approved]:
        result = generator.generate_explanation(case, top_k=3)
        print(f"--- {result['case_id']} ({result['decision']}) ---")
        print(f"Query: {result['query']}")
        print("\nRetrieved Policy Chunks:")
        for c in result["retrieved_chunks"]:
            print(f"  [{c['score']:.3f}] {c['source_file']}: {c['text'][:70]}...")
        print("\nGenerated Explanation (Mistral-7B):")
        print(result["explanation"])
        print("\n" + "=" * 66 + "\n")
