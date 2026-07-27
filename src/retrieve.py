"""
retrieve.py — Given a user query, return the most relevant chunks.

WHAT YOU NEED TO DECIDE (this is design, not boilerplate):
1. How many chunks to retrieve (top_k)? Too few = missing context.
   Too many = noisy, unfocused generation. Try a few values and note what
   you observed.
2. Do you do any post-retrieval filtering? e.g., deduplicate near-identical
   reviews, filter by a minimum similarity score threshold so junk doesn't
   get passed to generation.
3. What happens when NOTHING relevant is retrieved (e.g., a query about a
   product category you didn't index)? This matters a lot for hallucination
   prevention — if you pass empty/irrelevant context to the LLM anyway, it
   will likely make something up. Handle this case explicitly.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from src.embed_index import EMBED_MODEL_NAME, COLLECTION_NAME


def retrieve(query: str, persist_dir: str = "data/chroma_db", top_k: int = 5):
    model = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    # TODO: apply your filtering/threshold logic here before returning.
    # `results` contains documents, metadatas, and distances — decide what
    # "too irrelevant to use" means for your distance metric, and handle the
    # case where filtering leaves you with zero usable chunks.

    return results


if __name__ == "__main__":
    test_query = "is this good for gaming"  # replace with a real test query
    results = retrieve(test_query)
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"[{meta['product_title']}] {doc[:100]}...")
