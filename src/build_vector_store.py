"""
build_vector_store.py

Builds a FAISS vector index from the embeddings generated in Day 2, and
runs a quick sanity-check retrieval test to confirm the index actually
returns sensible results before we build the full retrieval module (Day 4).

Index choice: IndexFlatIP (inner product) over L2-normalized vectors,
which is equivalent to cosine similarity search. Flat (exhaustive) search
is the right choice here — our corpus is tiny (15 chunks), so there's no
need for approximate search structures like IVF or HNSW.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_data():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    return chunks, embeddings


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """L2-normalize embeddings, then build a flat inner-product index
    (inner product on normalized vectors = cosine similarity)."""
    normalized = embeddings.copy()
    faiss.normalize_L2(normalized)

    dim = normalized.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normalized)
    return index


def search(index: faiss.Index, model: SentenceTransformer, query: str, chunks: list[dict], top_k: int = 3):
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = chunks[idx]
        results.append({
            "score": float(score),
            "text": chunk["text"],
            "source_file": chunk["source_file"],
        })
    return results


def sanity_check(index: faiss.Index, model: SentenceTransformer, chunks: list[dict]):
    """Run a couple of realistic test queries and print what comes back,
    so we can eyeball whether retrieval is actually sensible before
    building the full retrieval module tomorrow."""
    test_queries = [
        "Applicant denied due to high debt-to-income ratio and short credit history",
        "What should a compliant credit denial explanation include?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = search(index, model, query, chunks, top_k=2)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['source_file']}: {r['text'][:100]}...")


def main():
    print("Loading chunks and embeddings...")
    chunks, embeddings = load_data()
    print(f"  {len(chunks)} chunks, embeddings shape {embeddings.shape}")

    print("\nBuilding FAISS index...")
    index = build_index(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"  Index saved to {INDEX_PATH} ({index.ntotal} vectors)")

    print("\nLoading embedding model for sanity-check queries...")
    model = SentenceTransformer(MODEL_NAME)

    print("\nRunning sanity-check retrieval tests...")
    sanity_check(index, model, chunks)


if __name__ == "__main__":
    main()
