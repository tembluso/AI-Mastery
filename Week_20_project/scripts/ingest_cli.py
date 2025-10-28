import argparse
from pathlib import Path

from rag.ingest import ingest_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest notes into FAISS index")
    ap.add_argument("--input_dir", type=str, required=True, help="Folder with PDFs/TXT/MD")
    args = ap.parse_args()

    total = ingest_dir(Path(args.input_dir))
    print(f"Done. {total} chunks indexed.")
