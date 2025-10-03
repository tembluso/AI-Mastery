
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# Optional PDF import (using pypdf)
try:
    from pypdf import PdfReader
    _HAS_PDF = True
except Exception:
    _HAS_PDF = False


@dataclass
class Passage:
    doc_id: str
    chunk_id: int
    text: str
    meta: Dict[str, str] = field(default_factory=dict)


def chunk_text(text: str, chunk_tokens: int = 180, overlap: int = 40) -> List[str]:
    """Simple whitespace token chunker with overlap."""
    toks = text.split()
    chunks: List[str] = []
    i = 0
    step = max(1, chunk_tokens - overlap)
    while i < len(toks):
        chunk = toks[i:i + chunk_tokens]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += step
    return chunks


def read_file_text(path: str) -> str:
    """Read .txt/.md/.pdf to plain text."""
    path = str(path)
    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if path.lower().endswith(".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if path.lower().endswith(".pdf"):
        if not _HAS_PDF:
            raise RuntimeError("pypdf not installed. Add 'pypdf' to requirements or upload .txt/.md instead.")
        reader = PdfReader(path)
        txt_parts: List[str] = []
        for p in reader.pages:
            try:
                txt_parts.append(p.extract_text() or "")
            except Exception:
                txt_parts.append("")
        return "\n".join(txt_parts)
    raise ValueError(f"Unsupported file type: {path}")


class RAGIndex:
    """
    Minimal RAG index with: chunking, SentenceTransformers embeddings, FAISS retrieval,
    optional CrossEncoder reranking, and a simple synthesis step.
    """
    def __init__(
        self,
        embed_model: str = "all-MiniLM-L6-v2",
        cross_encoder: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.embedder = SentenceTransformer(embed_model)
        self.cross_encoder_name = cross_encoder
        self.reranker: Optional[CrossEncoder] = None
        self.passages: List[Passage] = []
        self._faiss_index: Optional[faiss.IndexFlatL2] = None
        self._matrix: Optional[np.ndarray] = None
        self._dim: Optional[int] = None

        # Try loading reranker lazily
        if cross_encoder:
            try:
                self.reranker = CrossEncoder(cross_encoder)
            except Exception:
                self.reranker = None  # graceful fallback

    def ingest_files(self, file_paths: List[str], chunk_tokens: int = 180, overlap: int = 40) -> int:
        """Read files, chunk text, and collect passages (not yet embedded)."""
        count = 0
        for path in file_paths:
            text = read_file_text(path)
            doc_id = str(uuid.uuid4())[:8]
            chunks = chunk_text(text, chunk_tokens=chunk_tokens, overlap=overlap)
            for i, ch in enumerate(chunks):
                self.passages.append(Passage(doc_id=doc_id, chunk_id=i, text=ch, meta={"source": path}))
                count += 1
        return count

    def build(self) -> None:
        """Create FAISS index from current passages."""
        if not self.passages:
            raise RuntimeError("No passages ingested. Use ingest_files() first.")
        texts = [p.text for p in self.passages]
        mat = self.embedder.encode(texts, convert_to_numpy=True).astype("float32")
        self._dim = int(mat.shape[1])
        self._matrix = mat
        self._faiss_index = faiss.IndexFlatL2(self._dim)
        self._faiss_index.add(mat)

    def search(self, query: str, k: int = 5, rerank_top_m: int = 15, final_top_k: int = 3) -> List[Tuple[Passage, float]]:
        """FAISS search, then optional CrossEncoder rerank. Returns (passage, score)."""
        if self._faiss_index is None:
            raise RuntimeError("Index not built. Call build().")
        q = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self._faiss_index.search(q, max(k, rerank_top_m))
        cand_idxs = I[0].tolist()
        cands: List[Tuple[Passage, float]] = [(self.passages[i], float(D[0][j])) for j, i in enumerate(cand_idxs)]

        # Rerank if available (higher score = better)
        if self.reranker is not None and len(cands) > 0:
            pairs = [[query, p.text] for (p, _) in cands]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip([p for p, _ in cands], scores), key=lambda x: x[1], reverse=True)
            return [(p, float(s)) for p, s in ranked[:final_top_k]]
        else:
            # Fall back to FAISS L2 distances (lower is better) → convert to similarity proxy
            ranked = sorted(cands, key=lambda x: x[1])[:final_top_k]
            return [(p, 1.0 / (1e-6 + d)) for (p, d) in ranked]

    def synthesize(self, query: str, passages: List[Passage]) -> str:
        """
        Extractive synthesis:
        - Split retrieved passages into sentences
        - Embed sentences + query
        - Rank sentences by cosine similarity to the query
        - Return top N, deduped and trimmed
        """
        import re
        from sklearn.metrics.pairwise import cosine_similarity

        # Sentence splitter (nltk if available; else regex)
        try:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt")
            except Exception:
                nltk.download("punkt", quiet=True)
            from nltk.tokenize import sent_tokenize
            def split_sents(t): return sent_tokenize(t)
        except Exception:
            def split_sents(t): return re.split(r'(?<=[\.\!\?])\s+', t)

        # Collect candidate sentences
        sents = []
        for p in passages:
            for s in split_sents(p.text):
                s = s.strip()
                if 20 <= len(s) <= 240:   # filter too-short/too-long
                    sents.append(s)

        if not sents:
            return passages[0].text[:240] if passages else "No relevant context found."

        # Embed query + sentences
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        s_emb = self.embedder.encode(sents, convert_to_numpy=True)

        sims = cosine_similarity(q_emb, s_emb)[0]
        # Keep top N sentences with a soft threshold
        top_idx = sims.argsort()[::-1][:4]
        selected = []
        seen = set()
        for i in top_idx:
            if sims[i] < 0.35:   # similarity floor; bump to 0.4 if still noisy
                continue
            sent = sents[i]
            # simple dedup
            key = re.sub(r'\W+', '', sent.lower())
            if key not in seen:
                seen.add(key)
                selected.append(sent)

        if not selected:
            # Fallback: highest-scoring sentence
            selected = [sents[top_idx[0]]]

        # Join cleanly
        answer = " ".join(selected)
        # Tiny cleanup: collapse spaces
        answer = re.sub(r'\s+', ' ', answer).strip()
        return answer


    def answer(self, query: str, k: int = 3):
        """Search → synthesize → return answer with source previews."""
        hits = self.search(query, final_top_k=k)
        passages = [p for (p, _) in hits]
        answer = self.synthesize(query, passages)
        sources = [{
            "source": p.meta.get("source", ""),
            "chunk_id": p.chunk_id,
            "preview": p.text[:200],
        } for p in passages]
        return {"answer": answer, "sources": sources}
