"""
generate.py — Send retrieved context + query to an LLM via Groq's API.

WHAT YOU NEED TO DESIGN:
1. The prompt template. It needs to instruct the model to:
   - Only use the provided context (not its own general knowledge)
   - Cite which chunk/review each claim comes from
   - Say "I don't have enough information" if the context doesn't answer
     the question, rather than guessing
2. How you format the retrieved chunks into the prompt (numbered list?
   with product names? with source IDs so citations are traceable?)

This prompt is the single biggest lever on hallucination rate. Iterate on
it — try a few versions, and note in your README what you tried and what
worked better.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "llama-3.1-8b-instant"


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    TODO: design your prompt template here.

    retrieved_chunks is a list of dicts like:
        {"text": ..., "product_title": ..., "chunk_id": ...}

    Starting structure to adapt:

        You are answering questions using ONLY the context below.
        If the context does not contain the answer, say so explicitly.
        Cite the source chunk ID for every claim you make.

        Context:
        [1] (chunk_id_abc) Product: ... | Review: ...
        [2] (chunk_id_xyz) Product: ... | Review: ...

        Question: {query}

        Answer (with citations like [1], [2]):
    """
    prompt = ""  # YOUR IMPLEMENTATION HERE
    return prompt


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    prompt = build_prompt(query, retrieved_chunks)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Quick manual test once retrieve.py and this file are both implemented
    from src.retrieve import retrieve

    query = "is this good for gaming"
    results = retrieve(query)
    chunks = [
        {"text": doc, "product_title": meta["product_title"], "chunk_id": cid}
        for doc, meta, cid in zip(
            results["documents"][0], results["metadatas"][0], results["ids"][0]
        )
    ]
    answer = generate_answer(query, chunks)
    print(answer)