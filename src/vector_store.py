"""
vector_store.py — Lightweight flat vector store using NumPy.

WHY NOT CHROMADB: ChromaDB's dependency chain (via pandas) doesn't have
pre-built wheels for Python 3.13 on this machine, forcing a source build
that fails. Rather than fight the environment further, this uses a flat
NumPy-based cosine similarity search instead — a legitimate choice at this
data scale (~2,000 chunks), since brute-force search over that many vectors
is fast (milliseconds) and avoids the operational complexity of a vector
database that's designed for much larger scale (millions of vectors,
approximate nearest-neighbor indexing). At significantly larger scale,
ChromaDB or a similar ANN-based store would be the right call instead.
"""

import json
import numpy as np


class VectorStore:
    def __init__(self):
        self.ids: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.documents: list[str] = []
        self.metadatas: list[dict] = []

    def add(self, ids: list[str], embeddings: list[list[float]],
            documents: list[str], metadatas: list[dict]):
        self.ids = ids
        self.embeddings = np.array(embeddings, dtype=np.float32)
        self.documents = documents
        self.metadatas = metadatas

    def query(self, query_embedding: list[float], n_results: int = 5) -> dict:
        """Cosine similarity search — returns top n_results matches."""
        query_vec = np.array(query_embedding, dtype=np.float32)

        # Cosine similarity: dot product of normalized vectors
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        similarities = self.embeddings @ query_vec / (norms * query_norm + 1e-8)

        top_indices = np.argsort(similarities)[::-1][:n_results]

        return {
            "ids": [[self.ids[i] for i in top_indices]],
            "documents": [[self.documents[i] for i in top_indices]],
            "metadatas": [[self.metadatas[i] for i in top_indices]],
            "distances": [[float(1 - similarities[i]) for i in top_indices]],
        }

    def save(self, path: str = "data/vector_store.json"):
        data = {
            "ids": self.ids,
            "embeddings": self.embeddings.tolist(),
            "documents": self.documents,
            "metadatas": self.metadatas,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Saved vector store to {path}")

    @classmethod
    def load(cls, path: str = "data/vector_store.json") -> "VectorStore":
        with open(path, "r") as f:
            data = json.load(f)
        store = cls()
        store.ids = data["ids"]
        store.embeddings = np.array(data["embeddings"], dtype=np.float32)
        store.documents = data["documents"]
        store.metadatas = data["metadatas"]
        return store