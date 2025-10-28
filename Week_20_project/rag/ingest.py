from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
import json

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

from rag.config import INDEX_DIR, INDEX_PATH, META_PATH, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from rag.utils import load_document, make_chunks

load_dotenv()
client = OpenAI()

BATCH = 128

def _embed_texts(texts: List[str]) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    arr = np.array(vecs, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr

def _load_or_create_index(dim: int):
    if INDEX_PATH.exists():
        return faiss.read_index(str(INDEX_PATH))
    index = faiss.IndexFlatIP(dim)  # cosine via normalized inner product
    return index

def _append_meta(records: List[dict]):
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def iter_files(input_dir: Path) -> Iterable[Path]:
    for ext in ("*.pdf", "*.txt", "*.md", "*.markdown"):
        yield from Path(input_dir).rglob(ext)

def ingest_paths(paths: List[Path]):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    ids, texts, metas = [], [], []
    for p in paths:
        text = load_document(p)
        chunks = make_chunks(text, str(p), CHUNK_SIZE, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks):
            ids.append(f"{p.name}-{i}")
            texts.append(ch.text)
            metas.append({"source_path": str(p), "chunk_idx": i})

    if not texts:
        print("No texts found to index.")
        return 0

    vecs = _embed_texts(texts)
    dim = vecs.shape[1]
    index = _load_or_create_index(dim)
    index.add(vecs)
    faiss.write_index(index, str(INDEX_PATH))

    _append_meta([{"id": ids[i], "text": texts[i], "meta": metas[i]} for i in range(len(texts))])

    print(f"Indexed {len(texts)} chunks from {len(paths)} files → {INDEX_DIR}")
    return len(texts)

def ingest_dir(input_dir: Path):
    files = list(iter_files(input_dir))
    return ingest_paths(files)
