"""
ingest.py — Load and chunk product review data.

WHAT THIS FILE NEEDS TO DO:
1. Load your saved subset of the Amazon Reviews dataset
2. Clean the text (remove empty reviews, weird encoding, etc.)
3. Chunk the text into retrievable units

WHY CHUNKING STRATEGY MATTERS (read before you write this):
RAG quality depends heavily on chunk size and boundaries. If chunks are too big,
retrieval returns noisy, unfocused context. Too small, and you lose context
needed to answer the question. For product reviews specifically, consider:
  - Do you chunk per-review (each review = 1 chunk)?
  - Do you combine product metadata (title, category) with review text?
  - Do you split long reviews into smaller pieces, and if so, by sentence?
    by fixed token count? with overlap?

There's no single right answer — but you MUST be able to explain why you chose
what you chose. Write your reasoning as a comment above your implementation.
"""

import pandas as pd
import json
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    product_id: str
    product_title: str
    text: str
    source_type: str  # e.g. "review" or "product_description"


def load_raw_data(path: str) -> pd.DataFrame:
    """Load your saved CSV/parquet subset. Fill in the path and any cleaning."""
    df = pd.read_csv(path)

    # Drop rows with null or empty review text — an empty chunk is useless noise
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip().str.len() > 0]

    # Drop exact duplicate reviews (same text, same product) if any exist
    df = df.drop_duplicates(subset=["text", "parent_asin"])

    df = df.reset_index(drop=True)
    return df


import re

WORD_THRESHOLD = 150
OVERLAP_RATIO = 0.2

def split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter — good enough for review text, no NLP library needed."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def group_sentences_into_windows(sentences: list[str], max_words: int, overlap_ratio: float) -> list[str]:
    """Group sentences into ~max_words windows, with overlap between consecutive windows."""
    windows = []
    current_window = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_word_count + sentence_words > max_words and current_window:
            windows.append(" ".join(current_window))
            overlap_word_target = int(max_words * overlap_ratio)
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_window):
                overlap_count += len(s.split())
                overlap_sentences.insert(0, s)
                if overlap_count >= overlap_word_target:
                    break
            current_window = overlap_sentences
            current_word_count = sum(len(s.split()) for s in current_window)

        current_window.append(sentence)
        current_word_count += sentence_words

    if current_window:
        windows.append(" ".join(current_window))

    return windows


def chunk_reviews(df: pd.DataFrame) -> list[Chunk]:
    """
    Length-conditional chunking: reviews under WORD_THRESHOLD words stay as a
    single chunk. Longer reviews are split into sentence-grouped windows with
    ~20% overlap, since long reviews often cover multiple distinct topics
    (e.g. battery life, build quality, customer service in one review) and a
    single embedding would blur those together, hurting retrieval precision.

    Based on this dataset's actual distribution (median 36 words, 75th
    percentile 79 words), only ~10% of reviews exceed the 150-word threshold,
    so this keeps the common case simple and only adds complexity where it's
    actually needed.
    """
    chunks: list[Chunk] = []

    for idx, row in df.iterrows():
        text = str(row["text"]).strip()
        word_count = len(text.split())
        product_id = str(row["parent_asin"])  # adjust column name if needed
        product_title = str(row.get("title", ""))  # adjust if your title column differs

        if word_count <= WORD_THRESHOLD:
            chunks.append(Chunk(
                chunk_id=f"{product_id}_{idx}_0",
                product_id=product_id,
                product_title=product_title,
                text=text,
                source_type="review",
            ))
        else:
            sentences = split_into_sentences(text)
            windows = group_sentences_into_windows(sentences, WORD_THRESHOLD, OVERLAP_RATIO)
            for w_idx, window_text in enumerate(windows):
                chunks.append(Chunk(
                    chunk_id=f"{product_id}_{idx}_{w_idx}",
                    product_id=product_id,
                    product_title=product_title,
                    text=window_text,
                    source_type="review",
                ))

    return chunks

    import json

if __name__ == "__main__":
    df = load_raw_data("data/raw_reviews.csv")
    chunks = chunk_reviews(df)
    print(f"Loaded {len(df)} rows -> produced {len(chunks)} chunks")

    with open("data/chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__) + "\n")
    print("Saved chunks to data/chunks.jsonl")