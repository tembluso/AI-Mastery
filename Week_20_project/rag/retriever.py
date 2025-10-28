from __future__ import annotations
from typing import List, Dict
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

from rag.config import INDEX_PATH, META_PATH, EMBED_MODEL, TOP_K

load_dotenv()
client = OpenAI()

def _load_index():
    if not INDEX_PATH.exists():
        raise FileNotFoundError("FAISS index not found. Ingest documents first.")
    return faiss.read_index(str(INDEX_PATH))

def _load_meta() -> List[Dict]:
    if not META_PATH.exists():
        return []
    items = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items

def _embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array([resp.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(v)
    return v

def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    index = _load_index()
    meta = _load_meta()
    if not meta:
        return []
    qv = _embed_query(query)
    D, I = index.search(qv, min(top_k, len(meta)))
    out = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= len(meta):
            continue
        item = meta[idx]
        out.append({"id": item["id"], "text": item["text"], "meta": item.get("meta", {}), "score": float(score)})
    return out
