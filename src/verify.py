"""
verify.py — Hallucination check for generated answers.

APPROACH:
Rather than checking each claim against the best-matching chunk anywhere in
the retrieved set, this checks each claim specifically against the chunk(s)
it cites (e.g. a sentence ending in [1] is checked against retrieved_chunks[0]).
This catches "citation drift" — a claim that cites a source but doesn't
actually match that source's content closely, which a generic best-match
check would miss if the claim happened to resemble some other chunk instead.

Claims with no citation at all are flagged separately and never counted as
"supported," since the prompt design requires citations — an uncited claim
is a prompt-compliance failure independent of whether it happens to be
semantically similar to something in the context.

LIMITATIONS (worth naming explicitly):
- Embedding similarity catches semantic drift, but not subtler issues like
  a claim that correctly combines two true facts into a false implication
  ("battery lasts 8 hours" + "screen is bright" -> "bright screen for 8
  hours of gaming" might not be directly supported by either individual
  claim).
- Sentence-level granularity means a sentence with one true and one false
  sub-claim gets a single score, potentially averaging out the problem.
- With more time/budget: a cross-encoder reranker (scores query+chunk pairs
  jointly, more accurate than comparing separate embeddings) or a second
  LLM call acting as a "judge" would likely catch more subtle issues than
  cosine similarity alone.

  OBSERVED IN TESTING: in one test run, this citation-aware approach caught a
  case where the model attributed an identical claim to two different chunks
  ([1] and [5]) — only the correctly-attributed one scored as supported (0.647
  vs 0.389). This demonstrates the mechanism works when misattribution occurs,
  though generation is non-deterministic (temperature=0.2), so this specific
  failure mode doesn't reproduce on every run. The point isn't that
  misattribution always happens — it's that when it does, citation-aware
  checking catches it, whereas a generic best-match-anywhere check would not.

  OVERALL_SUPPORTED_RATIO OBSERVATIONS: across two test runs of the same query,
  overall_supported_ratio varied between 0.375 and 0.625 — largely driven by
  how many uncited summary/inferential sentences the model produced, not by
  actual hallucination differences. This is a limitation of sentence-level,
  citation-based scoring: it's sensitive to the model's phrasing style, not
  just factual grounding.

  # OBSERVED: a claim citing [1] with the accurate quote "awful" scored 0.47,
  just under threshold — manually verified the source text ("Sound quality
  is awful! Dont buy!") does contain the quoted claim accurately. The lower
  score likely came from the claim's wrapper phrasing ("Review [1]
  contradicts this by stating that...") not resembling the terse source
  text, even though the core fact was correctly cited. This suggests
  comparing only the quoted/core portion of a claim to the source — rather
  than the full sentence including meta-commentary — could reduce false
  negatives. Left as a known limitation / future improvement rather than
  implemented, given time constraints.
"""

import os
import re
import sys
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

sys.path.append(os.path.dirname(__file__))
from ingest import split_into_sentences
from embed_index import EMBED_MODEL_NAME

load_dotenv()
client = InferenceClient(token=os.environ.get("HF_TOKEN"))

MIN_SUPPORT_SIMILARITY = 0.5  # placeholder — tune against real examples, see notes below


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _extract_citations(sentence: str) -> list[int]:
    """Extract citation numbers like [1], [2] from a sentence."""
    return [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]


def _clean_sentence(sentence: str) -> str:
    """Strip citation markers before embedding, so '[1]' doesn't pollute the claim's meaning."""
    return re.sub(r"\[\d+\]", "", sentence).strip()


def verify_claims(generated_answer: str, retrieved_chunks: list[dict], threshold: float = MIN_SUPPORT_SIMILARITY) -> dict:
    sentences = split_into_sentences(generated_answer)

    chunk_texts = [c["text"] for c in retrieved_chunks]
    chunk_embeddings = [
        np.array(client.feature_extraction(t, model=EMBED_MODEL_NAME))
        for t in chunk_texts
    ] if chunk_texts else []

    claims = []
    supported_count = 0

    for sentence in sentences:
        cited_indices = _extract_citations(sentence)
        clean_text = _clean_sentence(sentence)
        if not clean_text:
            continue

        claim_embedding = np.array(client.feature_extraction(clean_text, model=EMBED_MODEL_NAME))

        valid_cited_indices = [i for i in cited_indices if 0 <= (i - 1) < len(chunk_embeddings)]

        if valid_cited_indices:
            scores = [
                _cosine_similarity(claim_embedding, chunk_embeddings[i - 1])
                for i in valid_cited_indices
            ]
            best_score = max(scores)
            supported = best_score >= threshold
        else:
            # No valid citation — never counted as supported, regardless of
            # similarity to unrelated chunks (see module docstring).
            scores = [_cosine_similarity(claim_embedding, emb) for emb in chunk_embeddings]
            best_score = max(scores) if scores else 0.0
            supported = False

        if supported:
            supported_count += 1

        claims.append({
            "text": clean_text,
            "cited_chunks": cited_indices,
            "supported": supported,
            "best_match_score": round(best_score, 3),
            "has_valid_citation": bool(valid_cited_indices),
        })

    overall_ratio = supported_count / len(claims) if claims else 0.0

    return {
        "claims": claims,
        "overall_supported_ratio": round(overall_ratio, 3),
    }


if __name__ == "__main__":
    from retrieve import retrieve
    from generate import generate_answer

    query = "is this good for gaming"
    results = retrieve(query)
    chunks = [
        {"text": doc, "product_title": meta["product_title"], "chunk_id": cid}
        for doc, meta, cid in zip(
            results["documents"][0], results["metadatas"][0], results["ids"][0]
        )
    ]
    answer = generate_answer(query, chunks)
    print("=== Generated Answer ===")
    print(answer)

    print("\n=== Verification (real answer) ===")
    verification = verify_claims(answer, chunks)
    for claim in verification["claims"]:
        status = "✓" if claim["supported"] else "✗"
        print(f"{status} [{claim['best_match_score']}] {claim['text'][:80]}")
    print(f"\nOverall supported ratio: {verification['overall_supported_ratio']}")

    # Deliberately fabricated claim — should score low / be flagged
    fake_answer = "This product can fly and has a built-in coffee maker [1]."
    print("\n=== Verification (fabricated claim, should be flagged) ===")
    fake_verification = verify_claims(fake_answer, chunks)
    for claim in fake_verification["claims"]:
        status = "✓" if claim["supported"] else "✗"
        print(f"{status} [{claim['best_match_score']}] {claim['text'][:80]}")