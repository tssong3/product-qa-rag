"""
verify.py — Hallucination check for generated answers.

This is your project's differentiator — most basic RAG tutorials stop at
generation. This file is intentionally left almost entirely to you, because
being able to explain this logic in depth is the most valuable single thing
this project can prove about you.

THE CORE IDEA (yours to implement):
For each claim/sentence in the generated answer, check whether it's actually
supported by the retrieved context. One reasonable approach:

  1. Split the generated answer into individual sentences/claims.
  2. For each claim, compute its embedding (reuse sentence-transformers).
  3. Compare it against the embeddings of the retrieved chunks using cosine
     similarity.
  4. If the best-matching chunk's similarity is below some threshold you
     choose, flag that claim as "unverified" or "low confidence."

QUESTIONS YOU SHOULD BE ABLE TO ANSWER ABOUT YOUR OWN IMPLEMENTATION:
- What threshold did you pick, and how did you pick it? (Try a few values
  against known-good and known-bad examples — this is your "evaluation
  methodology" story.)
- What are the failure modes of this approach? (Hint: embedding similarity
  catches semantic drift, but not subtler issues like a claim that combines
  two true facts into a false implication. It's fine to have limitations —
  just be able to name them.)
- What would you do differently with more time/budget? (e.g., use a
  stronger cross-encoder reranker instead of cosine similarity, or a
  second LLM call as a "judge")

Write your implementation below. Keep the function signature so app.py can
call it, but everything inside is yours.
"""

from sentence_transformers import SentenceTransformer, util

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def verify_claims(generated_answer: str, retrieved_chunks: list[dict], threshold: float = 0.5) -> dict:
    """
    Returns something like:
        {
            "claims": [
                {"text": "...", "supported": True, "best_match_score": 0.71},
                {"text": "...", "supported": False, "best_match_score": 0.31},
            ],
            "overall_supported_ratio": 0.5,
        }

    TODO: implement the logic described above.
    """
    result = {"claims": [], "overall_supported_ratio": 0.0}

    # YOUR IMPLEMENTATION HERE

    return result


if __name__ == "__main__":
    # Once implemented, test with a deliberately fabricated claim to confirm
    # your verification actually catches it.
    pass
