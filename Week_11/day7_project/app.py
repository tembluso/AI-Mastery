import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)

def token_to_vec(token: str, dim: int) -> np.ndarray:
    # Deterministic pseudo-embedding from token string
    h = abs(hash(token)) % (2**32)
    rng = np.random.RandomState(h)
    return rng.normal(0, 1, size=(dim,))

def build_embeddings(tokens, d_model, mode):
    if mode == "Numbers":
        # Map numbers directly to vectors via simple projection of scalar -> d_model
        # Using a fixed random weight for reproducibility
        set_seed(123)
        W_num = np.random.randn(1, d_model) / np.sqrt(1)
        x = np.array(tokens, dtype=float).reshape(-1,1) @ W_num  # (L, d_model)
    else:
        # Words: per-token deterministic vector
        x = np.stack([token_to_vec(t, d_model) for t in tokens], axis=0)  # (L, d_model)
    return x  # (L, d_model)

def init_params(d_model, num_heads, seed=42):
    set_seed(seed)
    W_q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
    W_k = np.random.randn(d_model, d_model) / np.sqrt(d_model)
    W_v = np.random.randn(d_model, d_model) / np.sqrt(d_model)
    W_o = np.random.randn(d_model, d_model) / np.sqrt(d_model)
    return W_q, W_k, W_v, W_o

def split_heads(x, num_heads):
    L, D = x.shape
    d_k = D // num_heads
    x = x.reshape(L, num_heads, d_k)                 # (L, H, d_k)
    x = np.transpose(x, (1,0,2))                     # (H, L, d_k)
    return x  # per-head first

def combine_heads(x):  # x: (H, L, d_k)
    H, L, d_k = x.shape
    return np.transpose(x, (1,0,2)).reshape(L, H*d_k)  # (L, D)

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q,K,V: (H, Lq, d_k), (H, Lk, d_k), (H, Lk, d_k)
    H, Lq, d_k = Q.shape
    Lk = K.shape[1]
    scores = np.matmul(Q, np.transpose(K, (0,2,1))) / np.sqrt(d_k)  # (H, Lq, Lk)

    if mask is not None:
        # mask shape should broadcast to (H, Lq, Lk). True=keep, False=mask
        scores = np.where(mask, scores, -1e9)

    # Stable softmax along last axis
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-9)  # (H, Lq, Lk)
    out = np.matmul(weights, V)  # (H, Lq, d_k)
    return out, weights

def make_mask(mask_type, Lq, Lk):
    if mask_type == "None":
        return None
    if mask_type == "Padding (last p as PAD)":
        # Example: last p keys are padding -> mask them off
        # We'll require Lq == Lk for simplicity
        p = max(0, Lk // 4)  # mask last quarter
        keep = np.ones((Lq, Lk), dtype=bool)
        if p > 0:
            keep[:, Lk-p:] = False
        return keep  # (Lq, Lk)
    if mask_type == "Causal (GPT-style)":
        # Allow attending only to <= position (Lq x Lk), assume Lq==Lk
        keep = np.triu(np.ones((Lq, Lk), dtype=bool), k=0)
        keep = keep.T  # transpose to get <= (row attends to columns <= row)
        # Actually for attention (queries are rows), we want keep[i,j]=True if j<=i
        keep = np.tril(np.ones((Lq, Lk), dtype=bool), k=0)
        return keep
    return None

def attention_forward(tokens, mode, d_model, num_heads, seed, mask_type):
    # Build embeddings
    X = build_embeddings(tokens, d_model, mode)  # (L, D)
    W_q, W_k, W_v, W_o = init_params(d_model, num_heads, seed=seed)

    # Linear projections
    Q = X @ W_q  # (L, D)
    K = X @ W_k
    V = X @ W_v

    # Reshape into heads
    H = num_heads
    d_k = d_model // H
    Qh = split_heads(Q, H)  # (H, L, d_k)
    Kh = split_heads(K, H)
    Vh = split_heads(V, H)

    # Mask
    L = X.shape[0]
    mask = make_mask(mask_type, L, L)
    if mask is not None:
        mask = np.broadcast_to(mask, (H, L, L))  # (H, Lq, Lk)

    # Attention
    Oh, weights = scaled_dot_product_attention(Qh, Kh, Vh, mask=mask)  # (H, L, d_k)

    # Combine heads and final proj
    O = combine_heads(Oh) @ W_o  # (L, D)

    return {
        "X": X, "Q": Q, "K": K, "V": V,
        "O": O, "weights": weights  # weights: (H, L, L)
    }

def plot_heatmap(ax, mat, title, xlabel="Keys", ylabel="Queries"):
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Attention Visualizer", layout="wide")
st.title("🧭 Attention Visualizer — Week 11 · Day 7")

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Input mode", ["Numbers", "Words"])
    default_text = "I love pizza" if mode == "Words" else "1, 4, 2, 7, 7, 3"
    user_input = st.text_input("Tokens (comma/space-separated)", value=default_text)
    d_model = st.slider("Model dim (d_model)", min_value=16, max_value=256, value=64, step=16)
    num_heads = st.slider("Heads", min_value=1, max_value=8, value=4, step=1)
    seed = st.number_input("Seed", value=42, step=1)
    mask_type = st.selectbox("Mask", ["None", "Padding (last p as PAD)", "Causal (GPT-style)"])
    go = st.button("Compute Attention")

def parse_tokens(text, mode):
    if mode == "Numbers":
        # split by comma or space
        raw = [t.strip() for t in text.replace(",", " ").split() if t.strip()]
        try:
            toks = [float(x) for x in raw]
        except Exception:
            toks = [0.0 for _ in raw]
        return toks
    else:
        # words: split by whitespace
        toks = [t for t in text.split() if t.strip()]
        return toks

tokens = parse_tokens(user_input, mode)

st.markdown("### Input tokens")
st.write(tokens)

if go:
    if len(tokens) == 0:
        st.warning("Please provide at least one token.")
    elif d_model % max(1, num_heads) != 0:
        st.error("d_model must be divisible by num_heads.")
    else:
        res = attention_forward(tokens, mode, d_model, num_heads, int(seed), mask_type)
        weights = res["weights"]  # (H, L, L)

        st.markdown("### Attention Weights (per head)")
        H, L, _ = weights.shape
        # Show up to 8 heads in rows of 4
        for h in range(H):
            if h % 4 == 0:
                cols = st.columns(min(4, H - h))
            with cols[h % 4]:
                fig, ax = plt.subplots()
                im = ax.imshow(weights[h], aspect="auto")
                ax.set_title(f"Head {h}")
                ax.set_xlabel("Keys")
                ax.set_ylabel("Queries")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)


        st.markdown("### Averaged Attention (across heads)")
        avg_w = weights.mean(axis=0)
        fig, ax = plt.subplots()
        im = ax.imshow(avg_w, aspect="auto")
        ax.set_title("Average over heads")
        ax.set_xlabel("Keys")
        ax.set_ylabel("Queries")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

        with st.expander("Show matrices (Q, K, V, Output)"):
            st.write("**Q**", res["Q"])
            st.write("**K**", res["K"])
            st.write("**V**", res["V"])
            st.write("**Output (after heads + W_o)**", res["O"])

st.caption("Tip: Try **Causal mask** and compare against **None** on a short sentence like “I love pizza”.")