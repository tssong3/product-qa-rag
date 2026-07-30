"""
app.py — Simple CLI to ask questions end-to-end.

Wires together retrieve -> generate -> verify. Shows the generated answer
alongside a per-claim verification breakdown, so it's clear not just
*whether* the answer is well-supported overall, but *which specific claims*
are backed by the retrieved reviews and which aren't.
"""

from retrieve import retrieve
from generate import generate_answer
from verify import verify_claims


def ask(query: str, top_k: int = 5):
    results = retrieve(query, top_k=top_k)

    if not results["documents"][0]:
        print("\nNo sufficiently relevant reviews found for this query.")
        print("(Retrieval threshold filtered out all results — see retrieve.py)")
        return

    chunks = [
        {"text": doc, "product_title": meta["product_title"], "chunk_id": cid}
        for doc, meta, cid in zip(
            results["documents"][0], results["metadatas"][0], results["ids"][0]
        )
    ]

    answer = generate_answer(query, chunks)
    verification = verify_claims(answer, chunks)

    print("\n--- ANSWER ---")
    print(answer)

    print("\n--- VERIFICATION ---")
    for claim in verification["claims"]:
        status = "✓ supported" if claim["supported"] else "✗ unsupported"
        citation_note = "" if claim["has_valid_citation"] else " (no citation)"
        print(f"[{claim['best_match_score']:.2f}] {status}{citation_note}")
        print(f"    {claim['text'][:100]}")

    ratio = verification.get("overall_supported_ratio", 0)
    supported_count = sum(1 for c in verification["claims"] if c["supported"])
    print(f"\nOverall supported ratio: {ratio:.2f} ({supported_count}/{len(verification['claims'])} claims)")
    print("\n--- SOURCES RETRIEVED ---")
    for i, chunk in enumerate(chunks, start=1):
        print(f"[{i}] {chunk['product_title']} (chunk_id: {chunk['chunk_id']})")


if __name__ == "__main__":
    print("Product Q&A RAG — type a question, or 'quit' to exit.\n")
    while True:
        q = input("Q: ").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        ask(q)
