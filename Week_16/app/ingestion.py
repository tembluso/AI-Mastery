# app/ingestion.py
import argparse
from pathlib import Path
import pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from config import DATA_DIR, VECTOR_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from utils import basic_clean, chunk_text

def read_text_files(root: Path):
    texts = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            texts[p.stem] = p.read_text(encoding="utf-8", errors="ignore").lower()
    return texts

def build_index(data_dir: Path, out_dir: Path, size: int, overlap: int, n_neighbors: int = 10):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_docs = read_text_files(data_dir)
    if not raw_docs:
        raise SystemExit(f"No .txt/.md files found in {data_dir}.")
    cleaned = {k: basic_clean(v) for k, v in raw_docs.items()}
    rows = []
    for doc_id, text in cleaned.items():
        for i, ch in enumerate(chunk_text(text, size, overlap)):
            rows.append({"doc_id": doc_id, "chunk_id": i, "text": ch})
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Chunking produced 0 chunks; check size/overlap.")
    vec = TfidfVectorizer(ngram_range=(1,2), max_df=0.9)
    X = vec.fit_transform(df["text"].tolist())
    nn = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors).fit(X)
    joblib.dump({"vectorizer": vec, "matrix": X, "nn": nn}, out_dir / "tfidf_index.joblib")
    df.to_parquet(out_dir / "chunks.parquet", index=False)
    q = "why use chunk overlap?"
    qv = vec.transform([q])
    dist, idx = nn.kneighbors(qv, n_neighbors=3)
    print(f"Index built - Docs={len(cleaned)} Chunks={len(df)}")
    print(df.iloc[idx[0]][['doc_id','chunk_id','text']])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out_dir", type=Path, default=VECTOR_DIR)
    ap.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = ap.parse_args()
    build_index(args.data_dir, args.out_dir, args.chunk_size, args.overlap)

if __name__ == "__main__":
    main()
