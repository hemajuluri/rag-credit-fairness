"""
embed_documents.py

Loads the chunks produced by chunk_documents.py and generates embeddings
for each one using a local sentence-transformers model (no API key needed,
runs on CPU). Saves the embeddings + chunk metadata for the vector store
build step (Day 3).

Model choice: all-MiniLM-L6-v2 — small (~80MB), fast on CPU, and a solid
baseline for short policy-text retrieval. Can be swapped for a larger model
later if retrieval quality needs it.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found — run chunk_documents.py first."
        )
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def embed_chunks(chunks: list[dict], model_name: str = MODEL_NAME) -> np.ndarray:
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def main():
    chunks = load_chunks()
    embeddings = embed_chunks(chunks)

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Saved to: {EMBEDDINGS_PATH}")
    print("\nNote: vector store (FAISS index) build + retrieval testing happens in Day 3.")


if __name__ == "__main__":
    main()
