# Week 15 · Day 7 — Mini-RAG Notebook (Streamlit App)

This is the **ship-ready project** for Week 15. It builds a small **Retrieval-Augmented Generation** pipeline that can answer questions grounded in uploaded **PDFs / TXT / Markdown**.

## Features
- **Ingestion** of PDFs/txt/md → text extraction
- **Chunking with overlap** (configurable)
- **Embeddings** via SentenceTransformers (`all-MiniLM-L6-v2`)
- **FAISS** retrieval (fast nearest neighbor search)
- **Optional reranking** with a CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
- **Grounded synthesis** (simple, local): composes answer from top passages
- **Streamlit UI**: upload files, build index, ask questions, see citations

> Runs on CPU. No external API keys required. Models are downloaded from Hugging Face the first time you run.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Acceptance Criteria (Week 15 Spec)
1. **Embedding + FAISS index** built from uploaded files.  
2. **Working retrieval + grounded answers** in the UI.  
3. **≥5 test queries** produce accurate, source-grounded outputs.  
4. **README** explains the pipeline and how to run.

### Notes
- If reranking download is slow or blocked, uncheck the **Enable reranking** checkbox to use FAISS ranking only.
- PDFs are parsed with `pypdf`; complex PDFs may extract with imperfect layout.
- The grounded answer uses a lightweight heuristic; swap in a local/API LLM if desired.
