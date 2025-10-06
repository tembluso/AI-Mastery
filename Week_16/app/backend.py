# app/backend.py
import re
import joblib, pandas as pd
from pathlib import Path
from config import (
    VECTOR_DIR, TOP_K, ANSWER_MODE,
    MAX_CONTEXT_CHARS, MAX_NEW_TOKENS, MIN_NEW_TOKENS,
    TEMPERATURE, TOP_P, NO_REPEAT_NGRAM_SIZE, REPETITION_PENALTY,
    MODEL_NAME, DEFAULT_DEPTH
)

# ---------- load vector store ----------
def load_vector_store(vector_dir: Path = VECTOR_DIR):
    obj = joblib.load(vector_dir / "tfidf_index.joblib")
    chunks = pd.read_parquet(vector_dir / "chunks.parquet")
    return obj["vectorizer"], obj["matrix"], obj["nn"], chunks

# ---------- retrieval ----------
def retrieve(query: str, k: int, vectorizer, nn, chunks: pd.DataFrame):
    q_vec = vectorizer.transform([query])
    dist, idx = nn.kneighbors(q_vec, n_neighbors=max(k, 3))
    res = chunks.iloc[idx[0]].copy()

    # De-duplicate per document (helps avoid repetitive headings)
    res["rank"] = range(1, len(res) + 1)
    res = res.sort_values("rank").drop_duplicates(subset=["doc_id"], keep="first")
    return res.head(k)

def build_extractive_answer(chunks_df: pd.DataFrame) -> str:
    parts = []
    for _, row in chunks_df.iterrows():
        parts.append(clean_snippet(row["text"])[:280])
    return "\n\n".join(parts)


TITLEY_PREFIX = re.compile(r"^(chapter|section|course notes|climate facts|introduction)\b[:\-]?\s*", re.I)

def clean_snippet(txt: str) -> str:
    return TITLEY_PREFIX.sub("", txt.strip())

def build_context(chunks_df: pd.DataFrame) -> tuple[str, list[dict]]:
    pieces, citations = [], []
    for i, (_, r) in enumerate(chunks_df.iterrows(), start=1):
        text = clean_snippet(r["text"])
        pieces.append(f"[{i}] {text}")
        citations.append({
            "ref": f"[{i}]",
            "doc_id": r["doc_id"],
            "chunk_id": int(r["chunk_id"]),
            "excerpt": text[:200]
        })
    ctx = "\n\n".join(pieces)
    if len(ctx) > MAX_CONTEXT_CHARS:
        ctx = ctx[:MAX_CONTEXT_CHARS]
    return ctx, citations

# ——— LLM pipeline (cached) ———
_generator = None
def get_generator():
    global _generator
    if _generator is None:
        from transformers import pipeline
        _generator = pipeline("text2text-generation", model=MODEL_NAME, device_map="auto")
    return _generator

PROMPT_TMPL = (
    "You are a helpful assistant. Use ONLY the provided context.\n"
    "• If the answer is not in the context, say: \"I don't know based on the provided documents.\"\n"
    "• Write in a {depth} style.\n"
    "• Cite chunk numbers inline like [1][2] when you use them.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

def build_generative_answer(chunks_df: pd.DataFrame, question: str, depth: str | None = None) -> tuple[str, list[dict]]:
    depth = (depth or DEFAULT_DEPTH).lower()
    ctx, citations = build_context(chunks_df)
    prompt = PROMPT_TMPL.format(context=ctx, question=question, depth=("detailed (4–6 sentences)" if depth=="detailed" else "concise (2–3 sentences)"))

    gen = get_generator()
    out = gen(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=MIN_NEW_TOKENS,
        do_sample=True,                 # <-- not greedy
        temperature=TEMPERATURE,
        top_p=TOP_P,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        repetition_penalty=REPETITION_PENALTY,
    )
    answer = out[0]["generated_text"].strip()
    return answer, citations

# ---------- public API ----------
def answer_query(query: str, k: int = TOP_K):
    vectorizer, matrix, nn, chunks = load_vector_store()
    retrieved = retrieve(query, k, vectorizer, nn, chunks)

    if ANSWER_MODE.lower() == "generative":
        answer, citations = build_generative_answer(retrieved, query)
    else:
        answer = build_extractive_answer(retrieved)
        citations = [{
            "ref": f"[{i+1}]",
            "doc_id": r["doc_id"],
            "chunk_id": int(r["chunk_id"]),
            "excerpt": clean_snippet(r["text"])[:200]
        } for i, (_, r) in enumerate(retrieved.iterrows())]

    return {"query": query, "answer": answer, "citations": citations}
