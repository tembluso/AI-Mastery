
import streamlit as st
from pathlib import Path
from typing import List
import json

from rag_index import RAGIndex

st.set_page_config(page_title="Mini-RAG (Week 15 · Day 7)", page_icon="📚")
st.title("📚 Mini-RAG — Week 15 · Day 7 Project")
st.caption("Upload PDFs or text files → build vector index → ask grounded questions with citations.")

with st.expander("How it works / Settings", expanded=False):
    st.markdown("""
**Pipeline:** Upload files → chunk (overlap) → embeddings (SentenceTransformers) → FAISS retrieval → (optional) CrossEncoder rerank → grounded synthesis.

- **Chunk tokens**: Approximate by words here (works well enough for PDFs/txt).
- **Overlap**: Keeps context continuity between chunks.
- **Reranker**: Uses a small cross-encoder if available; otherwise falls back to FAISS ranking.
""")

chunk_tokens = st.number_input("Chunk size (words)", min_value=50, max_value=600, value=180, step=10)
overlap = st.number_input("Overlap (words)", min_value=0, max_value=200, value=40, step=5)
use_reranker = st.checkbox("Enable reranking (CrossEncoder)", value=True)

uploaded = st.file_uploader("Upload .pdf / .txt / .md files", type=["pdf","txt","md"], accept_multiple_files=True)

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.files_saved = []

def save_uploaded_files(files) -> List[str]:
    saved: List[str] = []
    data_dir = Path(st.experimental_get_query_params().get("data_dir", ["/mnt/data/uploads"])[0])
    data_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        path = data_dir / f.name
        with open(path, "wb") as out:
            out.write(f.read())
        saved.append(str(path))
    return saved

col1, col2 = st.columns(2)
with col1:
    if st.button("🔧 Build / Rebuild Index", use_container_width=True, type="primary"):
        file_paths = save_uploaded_files(uploaded) if uploaded else []
        if not file_paths and not st.session_state.files_saved:
            st.error("Please upload at least one file.")
        else:
            if file_paths:
                st.session_state.files_saved.extend(file_paths)
            st.write("Files in index:")
            for p in st.session_state.files_saved:
                st.write("•", p)

            idx = RAGIndex(embed_model="all-MiniLM-L6-v2", cross_encoder=("cross-encoder/ms-marco-MiniLM-L-6-v2" if use_reranker else None))
            count = idx.ingest_files(st.session_state.files_saved, chunk_tokens=int(chunk_tokens), overlap=int(overlap))
            idx.build()
            st.session_state.index = idx
            st.success(f"Index built with {count} chunks.")
with col2:
    if st.button("🧪 Load Sample Data and Build", use_container_width=True):
        # Load packaged samples
        sample_dir = Path(__file__).parent / "sample_data"
        files = [str(p) for p in sample_dir.glob("*.txt")]
        st.session_state.files_saved = files
        idx = RAGIndex(embed_model="all-MiniLM-L6-v2", cross_encoder=("cross-encoder/ms-marco-MiniLM-L-6-v2" if use_reranker else None))
        count = idx.ingest_files(files, chunk_tokens=int(chunk_tokens), overlap=int(overlap))
        idx.build()
        st.session_state.index = idx
        st.success(f"Index built on {len(files)} files → {count} chunks.")

st.divider()

st.subheader("Ask a question")
query = st.text_input("Your question", placeholder="e.g., What does the document say about the Eiffel Tower?")

k = st.slider("Top-k passages", 1, 6, 3)

if st.button("🔎 Retrieve & Answer", disabled=(st.session_state.index is None or not query)):
    if st.session_state.index is None:
        st.error("Build the index first.")
    else:
        with st.spinner("Searching and synthesizing…"):
            result = st.session_state.index.answer(query, k=k)
        st.markdown("### ✅ Grounded Answer")
        st.write(result["answer"])

        st.markdown("### 📎 Sources")
        for i, src in enumerate(result["sources"], start=1):
            with st.expander(f"Source {i}: {src['source']}  —  chunk {src['chunk_id']}"):
                st.write(src["preview"] + ("…" if len(src["preview"])==200 else ""))

        st.download_button("⬇️ Download raw JSON result", data=json.dumps(result, indent=2), file_name="rag_result.json")

st.caption("Tip: If CrossEncoder model download is slow or fails, uncheck reranking to use FAISS-only ranking.")
