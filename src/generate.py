"""
generate.py — Send retrieved context + query to an LLM via Groq's API.

PROMPT DESIGN DECISIONS:
1. Numbered context blocks with chunk IDs, so the model can cite [1], [2]
   inline and those citations map back to real, traceable chunk_ids —
   this is what verify.py checks claims against later.
2. The "use only the provided context" and "say so if you don't know"
   instructions are stated twice — once up top, once right before the
   question — since models are more likely to follow instructions
   reinforced close to the generation point than ones stated once and
   effectively forgotten several paragraphs earlier.
3. If retrieve.py's similarity filtering returns zero chunks, this skips
   the LLM call entirely and returns a fixed refusal message. No point
   spending an API call asking the model to say "I don't know" when we
   already know there's nothing relevant to answer from.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "llama-3.1-8b-instant"

NO_CONTEXT_RESPONSE = (
    "I don't have enough information in the indexed reviews to answer that question."
)


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Builds a numbered-context prompt. Each chunk is labeled [1], [2], etc.,
    with its chunk_id and product title, so the model can cite sources and
    those citations can be checked against the actual retrieved chunks later.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        block = (
            f"[{i}] (chunk_id: {chunk['chunk_id']}) "
            f"Product: {chunk['product_title']}\n"
            f"Review: {chunk['text']}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    prompt = f"""You are answering a question about a product using ONLY the customer reviews provided below. Do not use any outside knowledge about this or similar products.

Rules:
- Every claim you make must be supported by one of the numbered reviews below, and cited using its number, e.g. [1], [2].
- If the reviews below do not contain enough information to answer the question, say so explicitly instead of guessing.

Context:
{context}

Question: {query}

Remember: only use the context above. If it doesn't answer the question, say you don't have enough information. Cite sources like [1], [2] for every claim.

Answer:"""

    return prompt


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return NO_CONTEXT_RESPONSE

    prompt = build_prompt(query, retrieved_chunks)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    from retrieve import retrieve

    test_queries = [
        "is this good for gaming",
        "how does this compare to a Tesla",  # should trigger the no-context path
    ]

    for query in test_queries:
        print(f"\n=== Query: {query} ===")
        results = retrieve(query)
        chunks = [
            {"text": doc, "product_title": meta["product_title"], "chunk_id": cid}
            for doc, meta, cid in zip(
                results["documents"][0], results["metadatas"][0], results["ids"][0]
            )
        ]
        answer = generate_answer(query, chunks)
        print(answer)