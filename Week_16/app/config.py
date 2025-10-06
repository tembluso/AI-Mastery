from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sample_docs"
VECTOR_DIR = ROOT / "vector_store"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 150
TOP_K = 3

# app/config.py
ANSWER_MODE = "generative"            # or "extractive"
MODEL_NAME = "google/flan-t5-small"   # try "google/flan-t5-base" for richer output
MAX_CONTEXT_CHARS = 3000
MAX_NEW_TOKENS = 220
MIN_NEW_TOKENS = 80                   # ensures it doesn’t stop at 1–2 words
TEMPERATURE = 0.3
TOP_P = 0.9
NO_REPEAT_NGRAM_SIZE = 3
REPETITION_PENALTY = 1.05
DEFAULT_DEPTH = "concise"             # "concise" or "detailed"
