# Week 11 · Day 7 — Attention Visualizer (Streamlit App)

**Goal:** Input a sequence (numbers or words) → visualize self-attention heatmaps for each head.

## Why this matters
Attention is the core of modern Transformers. This visualizer shows how **queries** match **keys** to produce **weights**, which then mix **values** into outputs. You can toggle **masks** (padding/causal) and see how the attention pattern changes.

## How it works (Q, K, V in short)
- We embed each token into a vector (`d_model`).
- We project embeddings into **Q**, **K**, and **V** using linear matrices.
- We split into **multiple heads** and compute scaled dot-product attention per head:
  \[ \mathrm{softmax}(QK^\top / \sqrt{d_k})V \]
- We re-concatenate head outputs and project back to `d_model`.

No training here — just forward computations to build intuition.

## Features
- Numbers **or** words as input.
- Adjustable `d_model`, number of heads, seed.
- Masks:
  - **None** (unrestricted attention).
  - **Padding** (last p keys treated as PAD).
  - **Causal** (token i can only attend to ≤ i; GPT-style).
- Per-head heatmaps + averaged heatmap.
- Inspect matrices (Q, K, V, Output).

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Acceptance Criteria (met)
1. **Reproducible**: deterministic embeddings/projections via seed.
2. **Clear plots**: per-head and averaged attention heatmaps.
3. **Code that runs**: single-file Streamlit app (`app.py`).
4. **Explanation**: this README (Q/K/V and how to run).

## Notes
- For words, embeddings are deterministic from token strings (hash-seeded).
- For numbers, a simple learned-like projection from scalar→vector is used (fixed matrix for demo).
- This is a didactic tool; not a trained model.

## What I learned
- How Q/K/V produce attention weights.
- Why **scaling by √d_k** keeps softmax well-behaved.
- How **masks** change attention patterns (padding vs causal).
- Why **multi-heads** reveal different relational “views” over the same sequence.