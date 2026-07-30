"""
embed_index.py — Embed chunks and load them into a vector store.

Note: embeddings are generated via Hugging Face's hosted Inference API
(hardware-driven — PyTorch dropped Intel macOS wheel support, so local
sentence-transformers isn't installable on this machine). The vector store
is a flat NumPy-based store rather than ChromaDB, also hardware-driven —
ChromaDB's dependency chain has no pre-built wheels for Python 3.13 on this
setup. Both are legitimate architecture choices at this project's scale,
documented here for transparency.
"""

import os
import json
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from vector_store import VectorStore

load_dotenv()
client = InferenceClient(token=os.environ.get("HF_TOKEN"))

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16


def load_chunks(path: str = "data/chunks.jsonl") -> list[dict]:
    chunks = []
    with open(path, "r") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for text in batch:
            embedding = client.feature_extraction(text, model=EMBED_MODEL_NAME)
            all_embeddings.append(embedding.tolist())
        print(f"Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
        time.sleep(0.5)
    return all_embeddings


def build_index(chunks: list[dict], persist_path: str = "data/vector_store.json"):
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    store = VectorStore()
    store.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "product_id": c["product_id"],
                "product_title": c["product_title"],
                "source_type": c["source_type"],
            }
            for c in chunks
        ],
    )
    store.save(persist_path)
    print(f"Indexed {len(chunks)} chunks")
    return store


if __name__ == "__main__":
    chunks = load_chunks("data/chunks.jsonl")
    build_index(chunks)