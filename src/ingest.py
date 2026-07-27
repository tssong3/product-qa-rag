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
    df = pd.read_csv(path)  # or pd.read_parquet if you saved parquet
    # TODO: drop nulls, empty text, duplicates as needed
    return df


def chunk_reviews(df: pd.DataFrame) -> list[Chunk]:
    """
    TODO: implement your chunking strategy here.

    Starting point suggestion (feel free to change):
    - One chunk per review if the review is short (<300 words)
    - Split longer reviews into paragraphs or fixed-size windows with ~20%
      overlap between windows so you don't cut a sentence's meaning in half

    Return a list of Chunk objects.
    """
    chunks: list[Chunk] = []

    # YOUR IMPLEMENTATION HERE

    return chunks


if __name__ == "__main__":
    df = load_raw_data("data/raw_reviews.csv")
    chunks = chunk_reviews(df)
    print(f"Loaded {len(df)} rows -> produced {len(chunks)} chunks")
