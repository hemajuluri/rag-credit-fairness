"""
chunk_documents.py

Reads the markdown documents in data/policy_docs/ and data/case_examples/,
splits them into retrieval-sized chunks, and saves the chunks (with source
metadata) to data/chunks.json for the embedding step to consume.

Chunking strategy: split on markdown headers (##) first, since our documents
are organized in clearly-titled sections that already make good retrieval
units. Falls back to paragraph splitting for any section that's still long.
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "chunks.json"
MAX_CHUNK_CHARS = 800  # keep chunks small enough to be precise on retrieval


def split_by_headers(text: str) -> list[str]:
    """Split markdown text on ## headers, keeping the header with its section."""
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    return [s.strip() for s in sections if s.strip()]


def split_long_section(section: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """If a section is still too long, split it further on paragraph breaks."""
    if len(section) <= max_chars:
        return [section]

    paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def chunk_file(filepath: Path) -> list[dict]:
    text = filepath.read_text(encoding="utf-8")
    sections = split_by_headers(text)
    chunks = []
    for section in sections:
        for sub_chunk in split_long_section(section):
            chunks.append({
                "text": sub_chunk,
                "source_file": filepath.name,
                "source_dir": filepath.parent.name,
            })
    return chunks


def build_all_chunks() -> list[dict]:
    all_chunks = []
    for subdir in ["policy_docs", "case_examples"]:
        folder = DATA_DIR / subdir
        for md_file in sorted(folder.glob("*.md")):
            file_chunks = chunk_file(md_file)
            all_chunks.extend(file_chunks)
            print(f"  {md_file.name}: {len(file_chunks)} chunks")
    return all_chunks


def main():
    print("Chunking documents...")
    chunks = build_all_chunks()

    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = i

    OUTPUT_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
