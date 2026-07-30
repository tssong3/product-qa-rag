"""
app_streamlit.py — Streamlit frontend for the Product Q&A RAG pipeline.

Run with: streamlit run src/app_streamlit.py (from the project root)

Wraps the same retrieve -> generate -> verify pipeline as app.py, with a
visual interface showing the answer, per-claim verification breakdown, and
retrieved sources.
"""

import streamlit as st
from retrieve import retrieve
from generate import generate_answer
from verify import verify_claims

st.set_page_config(page_title="Product Q&A RAG", page_icon="🔍", layout="centered")

st.title("🔍 Product Q&A RAG")
st.caption(
    "Ask a question about products in the indexed Amazon reviews dataset. "
    "Answers are grounded in retrieved reviews, with citations and a "
    "hallucination check on every claim."
)

query = st.text_input("Ask a question:", placeholder="e.g. Is this laptop good for gaming?")
top_k = st.slider("Number of reviews to retrieve", min_value=1, max_value=10, value=5)
submitted = st.button("Ask", type="primary")

if submitted and query.strip():
    with st.spinner("Retrieving relevant reviews..."):
        results = retrieve(query, top_k=top_k)

    if not results["documents"][0]:
        st.warning(
            "No sufficiently relevant reviews found for this query. "
            "(Retrieval threshold filtered out all results.)"
        )
    else:
        chunks = [
            {"text": doc, "product_title": meta["product_title"], "chunk_id": cid}
            for doc, meta, cid in zip(
                results["documents"][0], results["metadatas"][0], results["ids"][0]
            )
        ]

        with st.spinner("Generating answer..."):
            answer = generate_answer(query, chunks)

        with st.spinner("Verifying claims against sources..."):
            verification = verify_claims(answer, chunks)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Claim Verification")
        supported_count = sum(1 for c in verification["claims"] if c["supported"])
        total_claims = len(verification["claims"])
        ratio = verification.get("overall_supported_ratio", 0)

        st.metric(
            "Supported claims",
            f"{supported_count}/{total_claims}",
            f"{ratio:.0%} supported"
        )

        for claim in verification["claims"]:
            icon = "✅" if claim["supported"] else "⚠️"
            citation_note = "" if claim["has_valid_citation"] else " (no citation)"
            with st.expander(f"{icon} [{claim['best_match_score']:.2f}]{citation_note} {claim['text'][:80]}..."):
                st.write(claim["text"])
                st.caption(f"Best match score: {claim['best_match_score']:.3f} | Cited chunks: {claim['cited_chunks']}")

        st.subheader("Retrieved Sources")
        for i, chunk in enumerate(chunks, start=1):
            with st.expander(f"[{i}] {chunk['product_title']}"):
                st.write(chunk["text"])
                st.caption(f"chunk_id: {chunk['chunk_id']}")

elif submitted:
    st.warning("Please enter a question.")