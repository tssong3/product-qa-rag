"""
run_eval.py — Runs the full eval query set through the pipeline and prints
results in a format ready to paste into eval_queries.md.

Relevance judgment (Y/N) is still manual — that requires human review of
whether the answer actually makes sense. This script automates the
mechanical part: running each query and capturing the answer + verification
score, so you're not manually copy-pasting one query at a time.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from retrieve import retrieve
from generate import generate_answer
from verify import verify_claims

QUERIES = [
    ("Easy", "Is the sound quality good on this speaker?"),
    ("Easy", "Is this laptop good for gaming?"),
    ("Easy", "How is the battery life on this camera?"),
    ("Easy", "Does this SD card work well with a Galaxy phone?"),
    ("Easy", "Is this mouse good for the price?"),
    ("Easy", "Does this graphics card handle Skyrim well?"),
    ("Medium", "What are the downsides of this laptop?"),
    ("Medium", "Is this good for a teenager who plays casual games?"),
    ("Medium", "How does this compare to a more expensive alternative?"),
    ("Medium", "Is the charging speed fast on this device?"),
    ("Medium", "What do people dislike about this product?"),
    ("Medium", "Is this durable for daily use?"),
    ("Hard", "Is this good for professional use?"),
    ("Hard", "How does this perform in cold weather?"),
    ("Hard", "Is customer service responsive for this brand?"),
    ("Hard", "Does this work well for left-handed users?"),
    ("Trap", "How does this compare to a Tesla?"),
    ("Trap", "What's the best restaurant near this store?"),
    ("Trap", "Should I get this phone or a MacBook?"),
    ("Trap", "Is this covered by a lifetime warranty?"),
    ("Trap", "What's the return policy on Amazon?"),
    ("Trap", "Does this product cause cancer?"),
    ("Trap", "Suggest the best 3 phones for photography"),
    ("Trap", "What's the weather like today?"),
]


def run_eval():
    print("| # | Category | Query | Expected | Actual (summary) | Supported ratio | Notes |")
    print("|---|----------|-------|----------|-------------------|------------------|-------|")

    for i, (category, query) in enumerate(QUERIES, start=1):
        results = retrieve(query)

        if not results["documents"][0]:
            summary = "No context retrieved — refused"
            ratio = "N/A"
        else:
            chunks = [
                {"text": doc, "product_title": meta["product_title"], "chunk_id": cid}
                for doc, meta, cid in zip(
                    results["documents"][0], results["metadatas"][0], results["ids"][0]
                )
            ]
            answer = generate_answer(query, chunks)
            verification = verify_claims(answer, chunks)
            ratio = verification["overall_supported_ratio"]
            summary = answer.replace("\n", " ")[:100].replace("|", "-")

        expected = "Grounded answer" if category in ("Easy", "Medium") else (
            "Cautious/partial answer" if category == "Hard" else "Refusal / insufficient info"
        )

        print(f"| {i} | {category} | {query} | {expected} | {summary}... | {ratio} | |")


if __name__ == "__main__":
    run_eval()