
# Week 4 · Day 7 — Mini-Project: Movie Recommender 🎬

**Goal:** A small, ship‑able item/user‑based collaborative filtering app on MovieLens 100k.

## Files
- `app.py` — Streamlit app.
- `requirements.txt` — dependencies to run the app.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
By default, it will try to load local `u.data` and `u.item`. If not present, it fetches them from GroupLens.

## Features
- Item‑based or user‑based cosine similarity.
- Filter by minimum number of ratings per movie.
- Option to exclude already‑watched items.
- Optional minimum release year filter.
- Cached data loading and similarity computation for speed.

## What I learned
- How to construct a user–item matrix and compute cosine similarities.
- How to turn a notebook prototype into a tiny interactive app.
- Trade‑offs between item‑ and user‑based CF (scalability vs personalization).
