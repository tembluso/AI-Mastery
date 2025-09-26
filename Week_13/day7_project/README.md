# Week 13 · Day 7 — Mini GPT vs BERT Playground

Type a sentence → **GPT** continues it.  
Add `[MASK]` → **BERT** fills the blank.

## What you get
- **Streamlit app** (`app.py`) with two behaviors:
  - No `[MASK]` → GPT-2 continuation with adjustable sampling.
  - Contains `[MASK]` → BERT fill-mask with top‑5 candidates.
- **CLI** (`predict.py`) for quick tests in a terminal.
- **Zero training required** (uses pretrained `gpt2` and `bert-base-uncased`).

## Quickstart

### 1) Install deps
```bash
pip install -r requirements.txt
```

### 2) Run the app
```bash
streamlit run app.py
```
Then open the local URL shown in the terminal.

### 3) Try the CLI
```bash
python predict.py gpt "The future of AI is"
python predict.py bert "The capital of France is [MASK]." --top_k 5
```

## Why GPT vs BERT?
- **GPT (decoder-only)** is trained with **causal language modeling** → great for **generation**.
- **BERT (encoder-only)** is trained with **masked language modeling** → great for **understanding** (classification, embeddings) and **fill‑mask**.

## Files
- `app.py` — Streamlit UI.
- `predict.py` — CLI utility.
- `requirements.txt` — pinned minimal dependencies.

## Notes
- First run downloads model weights (~hundreds of MB). Subsequent runs use cache.
- CPU is fine; GPU speeds things up.
- For reproducibility, sampling is stochastic by design; set `temperature=0.7, top_k=40, top_p=0.9` for steadier outputs.
