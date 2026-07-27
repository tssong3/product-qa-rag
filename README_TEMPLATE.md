# Product Q&A RAG

*Replace this with a 2-3 sentence summary once it's built: what it does, in plain English.*

## Why I Built This

*A sentence or two on the motivation — e.g., grounding this in your KPMG RAG
recovery experience and wanting a from-scratch, fully-owned build.*

## Architecture

*A simple diagram or even just an ASCII flow works fine:*

```
User Query
   |
   v
Retrieval (ChromaDB + sentence-transformers)
   |
   v
Generation (Llama 3.1 via Ollama)
   |
   v
Verification (embedding-similarity hallucination check)
   |
   v
Answer + citations + confidence flags
```

## Design Decisions

*This is the most important section for interviews. For each of the following,
write 2-3 sentences: what you chose, and why.*

- **Chunking strategy:**
- **Embedding model choice:**
- **Retrieval top_k and filtering:**
- **Prompt design:**
- **Hallucination verification approach:**

## Evaluation

*Summarize your results from `eval/eval_queries.md` here — a few sentences
plus maybe a small table.*

## Limitations & What I'd Do With More Time

*Be honest here — this is a portfolio project, not production software, and
naming real limitations is a strength, not a weakness.*

## How to Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# pull the model
ollama pull llama3.1:8b

# prepare data (see GETTING_STARTED.md for dataset download instructions)
python -m src.ingest
python -m src.embed_index

# ask questions
python -m src.app
```

## Tech Stack

Python · ChromaDB · sentence-transformers · Ollama (Llama 3.1) · Amazon Reviews 2023 dataset

## Credits

*If you leaned on any tutorial/repo for structure, credit it here — one line, no shame in it.*
