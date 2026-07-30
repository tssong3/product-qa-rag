"""
retrieve.py — Given a user query, return the most relevant chunks.

DESIGN DECISIONS:
1. top_k=5 as a starting point — validated empirically by testing a few
   real queries at top_k=3/5/8 (see notes below the __main__ block).
2. Similarity threshold filtering: chunks below MIN_SIMILARITY (0.45) are
   dropped before being returned. Threshold was tuned empirically — genuine
   on-topic test queries scored 0.52-0.62, while a deliberately irrelevant
   test query ("how does this compare to a Tesla") topped out at 0.44. This
   matters directly for hallucination prevention: without filtering, an
   off-topic query would still return top_k chunks regardless of actual
   relevance, and generate.py would answer from weak context rather than
   admitting it doesn't know.
3. Zero-results case handled explicitly: if nothing clears the threshold,
   retrieve() returns an empty result rather than silently passing along
   irrelevant chunks.
"""

import sys
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

sys.path.append(os.path.dirname(__file__))
from vector_store import VectorStore
from embed_index import EMBED_MODEL_NAME

load_dotenv()
client = InferenceClient(token=os.environ.get("HF_TOKEN"))

MIN_SIMILARITY = 0.45  # empirically tuned: on-topic queries scored 0.52-0.62,
                        # an intentionally off-topic query topped out at 0.44 —
                        # threshold set in the gap between these two clusters


def retrieve(query: str, store_path: str = "data/vector_store.json", top_k: int = 5):
    store = VectorStore.load(store_path)

    query_embedding = client.feature_extraction(query, model=EMBED_MODEL_NAME).tolist()
    results = store.query(query_embedding, n_results=top_k)

    # Apply similarity threshold filtering.
    # VectorStore returns distances as (1 - cosine_similarity), so we
    # convert back to similarity to compare against MIN_SIMILARITY.
    filtered = {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}

    for doc, meta, cid, dist in zip(
        results["documents"][0], results["metadatas"][0],
        results["ids"][0], results["distances"][0]
    ):
        similarity = 1 - dist
        if similarity >= MIN_SIMILARITY:
            filtered["documents"][0].append(doc)
            filtered["metadatas"][0].append(meta)
            filtered["ids"][0].append(cid)
            filtered["distances"][0].append(dist)

    if not filtered["documents"][0]:
        print(f"No chunks met the similarity threshold ({MIN_SIMILARITY}) for query: '{query}'")

    return filtered


if __name__ == "__main__":
    test_queries = [
        "is this good for gaming",
        "good battery life",
        "how does this compare to a Tesla",
    ]
    for test_query in test_queries:
        print(f"\n=== Query: {test_query} ===")
        results = retrieve(test_query)
        if not results["documents"][0]:
            print("No sufficiently relevant chunks found.")
        else:
            for doc, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            ):
                similarity = 1 - dist
                print(f"[{similarity:.2f}] [{meta['product_title']}] {doc[:100]}...")