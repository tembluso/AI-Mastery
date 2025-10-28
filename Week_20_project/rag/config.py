from pathlib import Path
import os

# Paths
ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "vectorstore_faiss"
INDEX_DIR.mkdir(exist_ok=True, parents=True)
INDEX_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "meta.jsonl"   # JSONL: one object per vector (id, text, meta)
PROMPTS_DIR = ROOT / "prompts"

# Models (can be overridden by env vars or .env)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536-d
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Chunking
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 6

# Metadata keys
DOC_SOURCE = "source_path"
DOC_ID = "doc_id"
