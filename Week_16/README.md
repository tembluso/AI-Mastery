# 💬 Week 16 — Custom Q&A Bot (Final Demo)

This is the **Day 7** fully working app. It uses a simple **TF‑IDF retrieval** backend and an **extractive answer** (top chunks stitched together) so it runs fast on CPU with minimal dependencies.

## Quickstart

```bash
pip install -r requirements.txt
python -m app.ingestion --data_dir data/sample_docs --out_dir vector_store --chunk_size 750 --overlap 150
streamlit run app/frontend.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`).

## Structure
- `app/` → ingestion, backend, frontend, config, utils
- `data/sample_docs/` → example docs
- `vector_store/` → generated index files
- `requirements.txt`

**No external LLM required.**
