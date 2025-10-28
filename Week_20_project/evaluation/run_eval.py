from __future__ import annotations
import time, csv, json
from pathlib import Path
from typing import List, Dict

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

import os, sys
if "rag" not in sys.modules:
    sys.path.append(os.path.abspath("."))

from rag.config import EMBED_MODEL, CHAT_MODEL
from rag.retriever import retrieve
from evaluation.utils_eval import precision_at_k, recall_at_k, cosine_sim

load_dotenv()
client = OpenAI()

RESULTS_CSV = Path("evaluation") / "results.csv"

def embed_texts(texts: List[str]) -> np.ndarray:
    B = 128
    out = []
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return np.array(out, dtype=np.float32)

def embed_one(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    import numpy as _np
    return _np.array([resp.data[0].embedding], dtype=_np.float32)

def relevance_heuristic(text: str, keywords: List[str]) -> int:
    t = text.lower()
    for kw in keywords:
        if kw.lower() in t:
            return 1
    return 0

def evaluate_query(item: Dict, top_k: int = 6) -> Dict:
    q = item["query"]
    keywords = item.get("keywords", [])
    max_chars = item.get("max_chars", 260)

    t0 = time.time()
    hits = retrieve(q, top_k=top_k)
    t1 = time.time()
    retrieve_ms = (t1 - t0) * 1000

    if not hits:
        return {
            "query": q,
            "precision@k": 0.0,
            "recall@k": 0.0,
            "latency_ms_retrieve": retrieve_ms,
            "latency_ms_total": retrieve_ms,
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "cosine_sim_mean": 0.0,
            "generated": "",
        }

    texts = [h["text"] for h in hits]

    relevances = [relevance_heuristic(t, keywords) for t in texts]
    total_relevant = max(1, sum(relevances))
    prec = precision_at_k(relevances, top_k)
    rec = recall_at_k(relevances, total_relevant, top_k)

    context_blocks = []
    for h in hits:
        src = h.get("meta", {}).get("source_path", "unknown")
        context_blocks.append(f"[Source: {src}]\n{h['text']}")
    joined_context = "\n\n---\n\n".join(context_blocks)

    extras_text = ""
    user_msg = f"""Write 3 post variants for X/Twitter, each ≤ {max_chars} characters.
Ground EVERY line in the CONTEXT below. If context is weak/irrelevant, say so and stop.
{extras_text}

CONTEXT START
{joined_context}
CONTEXT END
"""
    system_prompt = "You write concise, faithful, grounded social posts. No fabrication; echo phrasing from context when possible."

    t2 = time.time()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
    )
    t3 = time.time()
    gen_ms = (t3 - t2) * 1000
    total_ms = retrieve_ms + gen_ms

    text_out = resp.choices[0].message.content
    usage = getattr(resp, "usage", None)
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    gen_vec = embed_one(text_out)
    chunk_vecs = embed_texts(texts)
    sims = []
    for i in range(len(texts)):
        sims.append(cosine_sim(gen_vec[0], chunk_vecs[i]))
    import numpy as _np
    mean_sim = float(_np.mean(sims)) if sims else 0.0

    return {
        "query": q,
        "precision@k": round(prec, 3),
        "recall@k": round(rec, 3),
        "latency_ms_retrieve": round(retrieve_ms, 2),
        "latency_ms_total": round(total_ms, 2),
        "tokens_prompt": prompt_tokens or 0,
        "tokens_completion": completion_tokens or 0,
        "cosine_sim_mean": round(mean_sim, 3),
        "generated": text_out.replace("\n", " ").strip(),
    }

def main():
    cfg = json.loads(Path("evaluation/queries.json").read_text(encoding="utf-8"))
    rows = []
    for item in cfg:
        res = evaluate_query(item, top_k=6)
        rows.append(res)
        print(f"[OK] {item['query']} -> P@k={res['precision@k']}  R@k={res['recall@k']}  sim={res['cosine_sim_mean']}  total_ms={res['latency_ms_total']}")

    RESULTS_CSV.parent.mkdir(exist_ok=True)
    header = ["query","precision@k","recall@k","latency_ms_retrieve","latency_ms_total","tokens_prompt","tokens_completion","cosine_sim_mean","generated"]
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        for r in rows: w.writerow({k:r.get(k,'') for k in header})

if __name__ == "__main__":
    main()
