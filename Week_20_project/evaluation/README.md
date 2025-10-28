# Evaluation Pack (Day 6)

This folder lets you **measure** retrieval, generation, efficiency, and grounding.

## Files
- `queries.json` — list of test queries with keywords for heuristic relevance.
- `run_eval.py` — runs end-to-end: retrieval → generation → metrics. Appends rows to `evaluation/results.csv`.
- `utils_eval.py` — metrics helpers (precision@k, recall@k, cosine).
- `score_labels.py` — optional: compute metrics from **human-labeled** CSV.

## Usage
From the **project root** (so imports resolve):
```bash
python -m evaluation.run_eval
```
Results land in `evaluation/results.csv`.

### Customize
Edit `evaluation/queries.json` to add your own queries and relevance keywords.
If you prefer human labels, create `evaluation/retrieval_labels.csv` and run:
```bash
python -m evaluation.score_labels --labels_csv evaluation/retrieval_labels.csv --k 6
```
