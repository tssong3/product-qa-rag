"""
app.py — Simple CLI to ask questions end-to-end.

This wires together retrieve -> generate -> verify. Once your other files
are implemented, this should just work. Feel free to later wrap this in
Streamlit (`streamlit run src/app_streamlit.py`) for a visual demo — a
working screenshot or short GIF in your README goes a long way.
"""

from src.retrieve import retrieve
from src.generate import generate_answer
from src.verify import verify_claims


def ask(query: str, top_k: int = 5):
    results = retrieve(query, top_k=top_k)

    if not results["documents"][0]:
        print("No relevant products/reviews found for this query.")
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
    print(f"Supported ratio: {verification.get('overall_supported_ratio', 'N/A')}")


if __name__ == "__main__":
    print("Product Q&A RAG — type a question, or 'quit' to exit.\n")
    while True:
        q = input("Q: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        ask(q)
