# RAG Tweet Assistant (Streamlit, FAISS)

**Goal**: Turn your personal notes (PDF/TXT/MD) into grounded social posts via a tiny RAG pipeline.

## Features
- Upload notes or pre-index a notes folder
- Local vector DB with **FAISS** (persisted under `./vectorstore_faiss`)
- Retrieval with top-k chunks → **OpenAI Chat** generates variants
- UI controls: platform, variants, character limit, hashtags, emojis

## How it works
1. **Ingest**: parse docs → sentence-aware chunking → **OpenAI embeddings** → store in **FAISS**; append metadata to `meta.jsonl`.
2. **Retrieve**: embed query → cosine search (inner product on normalized vectors) → return top-k chunks + metadata.
3. **Generate**: pass retrieved chunks + instructions to an LLM → post drafts.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # fill OPENAI_API_KEY
# Ingest your folder
python scripts/ingest_cli.py --input_dir ./data
# Run the UI
streamlit run app/streamlit_app.py
```

## Run Evaluation

```bash
python -m evaluation.run_eval

```

## Notes & Tips
- Tune `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K` in `rag/config.py`.
- Prefer **clean, structured** notes for better retrieval. Headings help.
- FAISS index path: `vectorstore_faiss/index.faiss`; metadata lives in `meta.jsonl` (one JSON per vector, aligned by order).

## Security
- Keys are read from `.env` or environment variables; never commit secrets.

## Roadmap (optional)
- Add reranking (bge-rerank) after FAISS search
- Add platform-specific templates (LinkedIn vs X)
- Add per-note tagging + filter by tag/date in UI
- Export to CSV/Notion
