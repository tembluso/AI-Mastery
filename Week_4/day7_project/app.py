
import re
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MovieLens Recommender", layout="centered")

@st.cache_data(show_spinner=False)
def load_movielens():
    data_path = Path(".")
    # Try local files first
    local_data = data_path.parent / "u.data"
    local_item = data_path.parent / "u.item"
    if local_data.exists() and local_item.exists():
        ratings = pd.read_csv(local_data, sep="\t", names=["user","item","rating","ts"], engine="python")
        titles  = pd.read_csv(local_item, sep="|", header=None, encoding="latin-1", usecols=[0,1], names=["item","title"])
    else:
        # Fallback to web (MovieLens 100k)
        base = "http://files.grouplens.org/datasets/movielens/ml-100k/"
        ratings = pd.read_csv(base + "u.data", sep="\t", names=["user","item","rating","ts"])
        titles  = pd.read_csv(base + "u.item", sep="|", header=None, encoding="latin-1", usecols=[0,1], names=["item","title"])
    return ratings, titles

@st.cache_data(show_spinner=False)
def build_matrices(ratings):
    R = ratings.pivot_table(index="user", columns="item", values="rating").fillna(0)
    item_sim = pd.DataFrame(cosine_similarity(R.T), index=R.columns, columns=R.columns)
    user_sim = pd.DataFrame(cosine_similarity(R), index=R.index, columns=R.index)
    return R, item_sim, user_sim

def extract_year(title: str):
    m = re.search(r"\((\d{4})\)", str(title))
    return int(m.group(1)) if m else None

def recommend_item_based(user_id, R, item_sim, titles, n=5, min_ratings=0, exclude_watched=True, min_year=None):
    user_ratings = R.loc[user_id]
    rated_mask = user_ratings > 0
    preds = {}
    # Support filters
    counts = ratings.groupby("item").size()
    for item in R.columns[~rated_mask if exclude_watched else slice(None)]:
        if min_ratings and counts.get(item, 0) < min_ratings:
            continue
        if min_year is not None:
            year = extract_year(titles.loc[titles.item == item, "title"].values[0])
            if year is None or year < min_year:
                continue
        sims = item_sim[item]
        denom = sims[rated_mask].sum()
        if denom == 0:
            continue
        preds[item] = float((user_ratings * sims).sum() / denom)
    top = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:n]
    results = []
    for iid, score in top:
        row = titles.loc[titles.item == iid, "title"]
        title = row.values[0] if not row.empty else str(iid)
        results.append((iid, title, score))
    return results

def recommend_user_based(user_id, R, user_sim, titles, n=5, min_ratings=0, exclude_watched=True, min_year=None):
    # Weighted average of neighbors' ratings
    sims = user_sim.loc[user_id]
    sims = sims.drop(user_id)  # remove self
    positive = sims[sims > 0]
    if positive.empty:
        st.warning("No positive-similarity neighbors; falling back to item-based might work better.")
    weights = positive / positive.sum() if positive.sum() != 0 else positive
    # Predicted rating for item j = sum_i w_i * R[i, j]
    preds_series = (weights.values[:, None] * R.loc[positive.index]).sum(axis=0)
    preds = {}
    counts = ratings.groupby("item").size()
    for item, score in preds_series.items():
        if exclude_watched and R.loc[user_id, item] > 0:
            continue
        if min_ratings and counts.get(item, 0) < min_ratings:
            continue
        if min_year is not None:
            title = titles.loc[titles.item == item, "title"].values[0]
            year = extract_year(title)
            if year is None or year < min_year:
                continue
        preds[item] = float(score)
    top = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:n]
    results = []
    for iid, score in top:
        title = titles.loc[titles.item == iid, "title"].values[0]
        results.append((iid, title, score))
    return results

# Load + build
ratings, titles = load_movielens()
R, item_sim, user_sim = build_matrices(ratings)

st.title("🎬 MovieLens Recommender (Week 4 Mini-Project)")

with st.sidebar:
    st.header("Controls")
    user_id = st.selectbox("User ID", sorted(R.index.tolist()), index=0)
    top_n = st.slider("Top N", 1, 15, 5)
    algo = st.radio("Algorithm", ["Item-based (cosine)", "User-based (cosine)"])
    min_r = st.slider("Min ratings per movie", 0, 200, 20, help="Filter out sparsely rated movies")
    exclude = st.checkbox("Exclude already-watched", True)
    year_filter = st.checkbox("Filter by min release year", False)
    min_year = st.number_input("Min year", 1900, 2025, 1990) if year_filter else None
    go = st.button("Recommend")

st.write("This demo uses the MovieLens 100k dataset. If the files 'u.data' and 'u.item' are found locally, "
         "they are used; otherwise they are fetched from GroupLens.")

if go:
    if algo.startswith("Item"):
        recs = recommend_item_based(user_id, R, item_sim, titles, n=top_n,
                                    min_ratings=min_r, exclude_watched=exclude, min_year=min_year)
    else:
        recs = recommend_user_based(user_id, R, user_sim, titles, n=top_n,
                                    min_ratings=min_r, exclude_watched=exclude, min_year=min_year)
    if not recs:
        st.info("No recommendations found with current filters. Try lowering 'min ratings' or turning off year filter.")
    else:
        st.subheader(f"Top {len(recs)} Recommendations for User {user_id}")
        out_df = pd.DataFrame(recs, columns=["item_id","title","score"])
        st.dataframe(out_df.style.format({"score": "{:.2f}"}), width="stretch")

st.caption("Tip: item-based CF scales well; user-based can be more personalized but heavier. Add metadata (genres/years) for hybrid models.")
