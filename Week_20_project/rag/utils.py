\
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import re
from pathlib import Path

from pypdf import PdfReader
from markdown_it import MarkdownIt

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

@dataclass
class DocChunk:
    text: str
    meta: Dict

def read_pdf(path: Path) -> str:
    pdf = PdfReader(str(path))
    parts = []
    for _, page in enumerate(pdf.pages):
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts)

def read_text_like(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in {".md", ".markdown"}:
        md = MarkdownIt()
        rendered = md.render(text)
        import re as _re
        return _re.sub(r"<[^>]+>", " ", rendered)
    return text

def load_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    elif ext in {".txt", ".md", ".markdown"}:
        return read_text_like(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def sentence_split(text: str) -> List[str]:
    sents = re.split(SENTENCE_RE, text)
    return [s.strip() for s in sents if s and s.strip()]

def make_chunks(text: str, source_path: str, chunk_size: int, overlap: int) -> List[DocChunk]:
    sents = sentence_split(text)
    chunks: List[DocChunk] = []
    buf: List[str] = []
    cur_len = 0

    for s in sents:
        if cur_len + len(s) > chunk_size and buf:
            chunk_text = " ".join(buf).strip()
            chunks.append(DocChunk(text=chunk_text, meta={"source_path": source_path}))
            # start overlap
            overlap_s = []
            ol_len = 0
            for rs in reversed(buf):
                if ol_len + len(rs) <= overlap:
                    overlap_s.insert(0, rs)
                    ol_len += len(rs)
                else:
                    break
            buf = overlap_s + [s]
            cur_len = sum(len(x) for x in buf)
        else:
            buf.append(s)
            cur_len += len(s)

    if buf:
        chunk_text = " ".join(buf).strip()
        chunks.append(DocChunk(text=chunk_text, meta={"source_path": source_path}))

    return chunks
