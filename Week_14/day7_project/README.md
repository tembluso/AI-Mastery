# Week 14 · Day 7 — Fine-Tuned LLM Demo (CPU-Friendly)

This mini-project ships a Streamlit app for **sentiment classification** (DistilBERT) and optional **review-style generation** (GPT-2).
It is designed for **CPU only** and avoids `Trainer` features that older `transformers` versions might not support.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

By default the app expects checkpoints at:
- `./finetuned-distilbert-imdb`
- `./finetuned-gpt2-imdb` (optional for the generation tab)

If missing, open the app's **Sidebar** and click **Bootstrap (tiny, CPU)** to train a *very small* model quickly
(e.g., 300–800 samples, 1 epoch). This is just to create a runnable demo on CPU — accuracy will be modest.

## Reproducible Training (offline, still CPU-friendly)
```bash
# Classification
python train_cls.py --epochs 1 --train_samples 4000 --eval_samples 1000 --model distilbert-base-uncased --outdir ./finetuned-distilbert-imdb

# Generation (optional, slower on CPU)
python train_gen.py --epochs 1 --train_samples 2000 --eval_samples 500 --model gpt2 --outdir ./finetuned-gpt2-imdb
```

## Files
- `app.py` — Streamlit app (inference-first; optional tiny bootstrap inside the app).
- `train_cls.py` — Simple PyTorch loop for DistilBERT classification (no `Trainer`).
- `train_gen.py` — Simple PyTorch loop for GPT-2 causal LM (no `Trainer`).
- `predict.py` — CLI sentiment prediction using the saved classifier.
- `generate.py` — CLI text generation using the saved generator.

> Notes: CPU only, minimal samples/epochs by default. Increase samples/epochs offline to improve metrics.
