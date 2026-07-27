"""
embed_index.py — Embed chunks and load them into ChromaDB.

This file is mostly plumbing (mechanical, fine to use as-is). The one thing
worth understanding well enough to explain: WHY this embedding model, and
what its limitations are (e.g., MiniLM is fast and free but less accurate
than larger models — that's a real tradeoff you made, be ready to say so).
"""

import chromadb
from sentence_transformers import SentenceTransformer
from src.ingest import Chunk

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, free, fast — good enough to start
COLLECTION_NAME = "product_reviews"


def build_index(chunks: list[Chunk], persist_dir: str = "data/chroma_db"):
    model = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=persist_dir)

    # Fresh collection each run — fine for a portfolio project;
    # in production you'd think about incremental updates instead.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"product_id": c.product_id, "product_title": c.product_title, "source_type": c.source_type}
            for c in chunks
        ],
    )
    print(f"Indexed {len(chunks)} chunks into '{COLLECTION_NAME}'")
    return collection


if __name__ == "__main__":
    from src.ingest import load_raw_data, chunk_reviews

    df = load_raw_data("data/raw_reviews.csv")
    chunks = chunk_reviews(df)
    build_index(chunks)
