"""
retrieval.py

A clean, reusable retrieval module — turns the Day 3 standalone script into
an importable component the generation module (Day 6-7) can actually use.

Two pieces:
1. DocumentRetriever — loads the FAISS index + embedding model once,
   exposes a simple .retrieve(query, top_k) method.
2. build_query_from_case() — takes an applicant's SHAP feature attributions
   (the actual input this system will receive in production) and turns them
   into a natural-language query string, since our real queries won't be
   free-text questions — they'll be derived from a model decision.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
MODEL_NAME = "all-MiniLM-L6-v2"


class DocumentRetriever:
    """Wraps the FAISS index + embedding model behind a simple retrieve() call.
    Loads everything once at construction so repeated queries are fast."""

    def __init__(self, index_path: Path = INDEX_PATH, chunks_path: Path = CHUNKS_PATH, model_name: str = MODEL_NAME):
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(index_path))
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the top_k most relevant chunks for a query, each with
        text, source_file, and a similarity score (higher = more relevant)."""
        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 if fewer than top_k results exist
                continue
            chunk = self.chunks[idx]
            results.append({
                "score": float(score),
                "text": chunk["text"],
                "source_file": chunk["source_file"],
            })
        return results


def build_query_from_case(top_features: list[dict], decision: str = "denied") -> str:
    """Turn an applicant's top SHAP feature attributions into a natural-
    language retrieval query.

    top_features: list of dicts like
        [{"feature": "debt-to-income ratio", "value": "0.42", "direction": "high"}, ...]
    decision: string indicating the application outcome, e.g. "approved" or "denied".

    This is what the real Day 6-7 generation pipeline will call before
    retrieving context for an actual applicant's explanation.
    """
    if not top_features:
        raise ValueError("top_features cannot be empty — need at least one factor to build a query")

    descriptions = []
    for feat in top_features:
        direction = feat.get("direction", "")
        descriptions.append(f"{direction} {feat['feature']}".strip())

    return f"Applicant {decision.lower()} due to: {', '.join(descriptions)}"


if __name__ == "__main__":
    # Quick manual check that the module works end to end when run directly.
    retriever = DocumentRetriever()

    # Simulate a query built from Case 1's SHAP features (from sample_cases.md)
    query = build_query_from_case(
        [
            {"feature": "debt-to-income ratio", "value": "0.42", "direction": "high"},
            {"feature": "credit history length", "value": "14 months", "direction": "short"},
        ],
        decision="denied",
    )
    print(f"Built query: {query}\n")

    results = retriever.retrieve(query, top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['source_file']}: {r['text'][:100]}...")

