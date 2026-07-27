# Getting Started — Product Q&A RAG

This is a scaffold, not a finished project. Files marked with `# TODO` contain the
design decisions you need to make and implement yourself — that's the part an
interviewer will actually ask you about, so don't skip it or copy it blindly.

The plumbing (installing libraries, connecting to ChromaDB, calling Groq) is
filled in for you, since that's boilerplate no one expects you to reinvent.

---

## Step 0 — Environment Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 1 — Set Up Groq API Access

> Note: this project uses Groq's hosted API instead of a local LLM (Ollama),
> since local inference wasn't practical on the development machine used.
> This is a legitimate architecture choice, not a shortcut — worth mentioning
> in your README's "design decisions" section.

Sign up at [console.groq.com](https://console.groq.com), then go to
**API Keys** and create a new key — copy it somewhere safe, you won't be able
to view it again.

Create a `.env` file in your project root:
GROQ_API_KEY=your_key_here

Make sure `.env` is listed in `.gitignore` **before** you commit anything —
check with `git status` that `.env` doesn't show up as a trackable file.

Install the packages this needs:

```bash
pip install groq python-dotenv
```

And add both to `requirements.txt`:

## Step 2 — Get the Data

Go to https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

Pick ONE category to start (recommendation: something you'd actually shop for —
you'll write better test queries later if you understand the products).
Categories like "Electronics" or "Software" tend to have cleaner text.

```python
from datasets import load_dataset
# Example — adjust the category name to what you picked
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                        "raw_review_Electronics",
                        split="full",
                        trust_remote_code=True)
```

Save a subset (a few thousand rows is plenty — you don't need millions) to
`data/raw_reviews.csv` or `.parquet` before moving on. Don't commit the full
dataset to GitHub — add `data/` to `.gitignore` and instead document in your
README exactly how to regenerate it.

## Step 3 — Work Through `src/` in Order

1. `src/ingest.py` — clean and chunk the data (**TODO: your chunking strategy**)
2. `src/embed_index.py` — embed chunks and load into ChromaDB (mostly done for you)
3. `src/retrieve.py` — retrieval function (**TODO: your retrieval logic/filtering**)
4. `src/generate.py` — prompt Ollama with retrieved context (**TODO: your prompt design**)
5. `src/verify.py` — hallucination check (**TODO: entirely yours — this is your differentiator**)
6. `src/app.py` — simple CLI loop to ask questions end-to-end

## Step 4 — Evaluate

Fill in `eval/eval_queries.md` with 20-30 real test queries and score them
honestly. This is the part most portfolio projects skip, and it's a strong
signal when you have it.

## Step 5 — Write the Real README

Replace this file's role with a proper `README.md` once the project works —
use `README_TEMPLATE.md` as your starting structure.
