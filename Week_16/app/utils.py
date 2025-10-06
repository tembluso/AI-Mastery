import re
def basic_clean(s: str) -> str:
    s = s.replace("\r"," ").replace("\n"," ")
    s = re.sub(r"\s+"," ", s).strip()
    return s
def chunk_text(text: str, size: int, overlap: int):
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text): break
        start = end - overlap
    return chunks
