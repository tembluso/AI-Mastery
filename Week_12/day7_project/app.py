# app.py
# Mini-Transformer Demo — Week 12 · Day 7
# Streamlit UI: train on toy reverse task, evaluate, predict, and visualize cross-attention.

import numpy as np  # import NumPy FIRST to avoid rare Windows init bug
import streamlit as st
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# Global config
# -----------------------
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Data
# -----------------------
def make_data(num_samples=2000, seq_len=10, vocab_size=20):
    X, Y = [], []
    for _ in range(num_samples):
        seq = torch.randint(1, vocab_size, (seq_len,))
        X.append(seq)
        Y.append(torch.flip(seq, dims=[0]))  # reverse task
    return torch.stack(X), torch.stack(Y)

@st.cache_resource(show_spinner=False)
def get_dataset(vocab_size=20, seq_len=10, num_samples=2000):
    X, Y = make_data(num_samples, seq_len, vocab_size)
    split = int(0.8 * num_samples)
    train_X, train_Y = X[:split], Y[:split]
    test_X,  test_Y  = X[split:],  Y[split:]
    return train_X, train_Y, test_X, test_Y

# -----------------------
# Model
# -----------------------
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=128, num_layers=2, max_len=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_emb = nn.Embedding(vocab_size, d_model)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.enc = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model, num_heads, d_ff, batch_first=True)
             for _ in range(num_layers)]
        )
        self.dec = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model, num_heads, d_ff, batch_first=True)
             for _ in range(num_layers)]
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, tgt_mask=None):
        src_e = self.src_emb(src) + self.pe[:, :src.size(1)]
        tgt_e = self.tgt_emb(tgt) + self.pe[:, :tgt.size(1)]
        mem = src_e
        for layer in self.enc:
            mem = layer(mem)
        out = tgt_e
        for layer in self.dec:
            out = layer(out, mem, tgt_mask=tgt_mask)
        return self.fc(out)

def causal_mask(T, device=DEVICE):
    m = torch.triu(torch.ones(T, T, device=device), diagonal=1)
    return m.masked_fill(m == 1, float("-inf"))

# For attention viz (since TransformerDecoderLayer doesn’t expose weights)
class DecoderBlockWithWeights(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x, memory, tgt_mask=None, need_weights=False):
        z, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + z)
        z, w = self.cross_attn(x, memory, memory, need_weights=need_weights, average_attn_weights=False)
        x = self.norm2(x + z)
        x = self.norm3(x + self.ff(x))
        return x, w

# @torch.no_grad()
# def get_cross_attn_weights(model, src, tgt_inp, tgt_mask=None, device=DEVICE):
#     model.eval()
#     src_e = model.src_emb(src) + model.pe[:, :src.size(1)]
#     mem = src_e.to(device)
#     for layer in model.enc:
#         mem = layer(mem)

#     any_dec = model.dec[0]
#     d_model = mem.size(-1)
#     num_heads = any_dec.self_attn.num_heads
#     # access FF dim safely:
#     try:
#         d_ff = any_dec.linear1.out_features
#     except:
#         d_ff = d_model * 2
#     viz = DecoderBlockWithWeights(d_model, num_heads, d_ff).to(device).eval()

#     tgt_e = model.tgt_emb(tgt_inp) + model.pe[:, :tgt_inp.size(1)]
#     _, w = viz(tgt_e.to(device), mem, tgt_mask=tgt_mask, need_weights=True)
#     return w  # shape: [B,H,T,S] or [H,T,S] depending on torch version

@torch.no_grad()
def get_cross_attn_weights_trained(model, src, tgt_inp, tgt_mask=None, device=DEVICE):
    """
    Returns cross-attention weights from the FIRST trained decoder layer.
    Shapes: src=[B,S], tgt_inp=[B,T]
    """
    model.eval()
    src = src.to(device); tgt_inp = tgt_inp.to(device)

    # 1) Encode with trained encoder
    src_e = model.src_emb(src) + model.pe[:, :src.size(1)]
    mem = src_e
    for enc in model.enc:
        mem = enc(mem)  # [B,S,d]

    # 2) Build decoder inputs up to *just before* cross-attn using the trained layer parts
    dec0 = model.dec[0]  # first trained decoder layer
    tgt_e = model.tgt_emb(tgt_inp) + model.pe[:, :tgt_inp.size(1)]
    # masked self-attn (uses trained dec0.self_attn)
    self_out, _ = dec0.self_attn(tgt_e, tgt_e, tgt_e, attn_mask=tgt_mask)
    x = dec0.norm1(tgt_e + self_out)

    # 3) Cross-attn with NEED WEIGHTS from the trained module
    #    average_attn_weights=False -> keep per-head weights
    cross_out, W = dec0.multihead_attn(
        x, mem, mem, need_weights=True, average_attn_weights=False
    )
    # Continue normally (not needed for weights)
    # x = dec0.norm2(x + cross_out); x = dec0.norm3(x + dec0.linear2(dec0.dropout(dec0.activation(dec0.linear1(x)))))

    # W can be [B,H,T,S] (new) or [T,B,S] etc. We’ll just return it.
    return W


def plot_attn(W, title):
    if W.dim() == 4:
        W = W[0, 0]  # first batch, first head
    elif W.dim() == 3:
        W = W[0]     # first head
    fig, ax = plt.subplots()
    ax.imshow(W.cpu(), aspect="auto")
    ax.set_xlabel("Encoder positions")
    ax.set_ylabel("Decoder positions")
    ax.set_title(title)
    st.pyplot(fig)

# -----------------------
# Training / Eval
# -----------------------
def train(model, train_X, train_Y, epochs=12, lr=1e-3, batch_size=128):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(train_X, train_Y)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    for ep in range(1, epochs + 1):
        total = 0.0
        for Xb, Yb in dl:
            Xb = Xb.to(DEVICE)
            tgt_inp = Yb[:, :-1].to(DEVICE)
            tgt_out = Yb[:, 1:].to(DEVICE)
            mask = causal_mask(tgt_inp.size(1))

            logits = model(Xb, tgt_inp, tgt_mask=mask)
            loss = crit(logits.reshape(-1, model.vocab_size), tgt_out.reshape(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        st.write(f"Epoch {ep:02d}/{epochs} — loss {total/len(dl):.4f}")

@torch.no_grad()
def evaluate(model, X, Y):
    model.eval()
    tgt_inp = Y[:, :-1].to(DEVICE)
    tgt_out = Y[:, 1:].to(DEVICE)
    mask = causal_mask(tgt_inp.size(1))
    logits = model(X.to(DEVICE), tgt_inp, tgt_mask=mask)
    preds = logits.argmax(-1)

    acc = (preds == tgt_out).float().mean().item()

    # Simple BLEU-1 (unigram precision)
    matches, total = 0, 0
    for p, ref in zip(preds, tgt_out):
        ref_set = set(ref.tolist())
        for tok in p.tolist():
            matches += int(tok in ref_set)
            total += 1
    bleu1 = matches / max(total, 1)
    return acc, bleu1, preds

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Mini-Transformer Demo", layout="centered")
st.title("Week 12 · Day 7 — Mini-Transformer Demo")
st.caption("Tiny encoder–decoder on a reverse-numbers toy task, with attention visualization.")

# Sidebar controls
with st.sidebar:
    st.header("Training Settings")
    epochs = st.slider("Epochs", 4, 30, 12, 1)
    d_model = st.select_slider("d_model", options=[32, 48, 64, 96], value=64)
    num_layers = st.select_slider("Layers", options=[1, 2, 3], value=2)
    num_heads = st.select_slider("Heads", options=[1, 2, 4, 8], value=4)
    d_ff = d_model * 2
    batch_size = st.select_slider("Batch size", options=[64, 128, 256], value=128)

train_X, train_Y, test_X, test_Y = get_dataset()
if "model" not in st.session_state:
    st.session_state.model = TinyTransformer(vocab_size=20, d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers).to(DEVICE)

col1, col2 = st.columns(2)
with col1:
    if st.button("Train / Retrain"):
        st.session_state.model = TinyTransformer(vocab_size=20, d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers).to(DEVICE)
        train(st.session_state.model, train_X, train_Y, epochs=epochs, batch_size=batch_size)
        torch.save(st.session_state.model.state_dict(), "checkpoint.pt")
        st.success("Training complete. Saved to checkpoint.pt")

with col2:
    if st.button("Load checkpoint.pt"):
        st.session_state.model = TinyTransformer(vocab_size=20, d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers).to(DEVICE)
        try:
            st.session_state.model.load_state_dict(torch.load("checkpoint.pt", map_location=DEVICE))
            st.success("Loaded checkpoint.pt")
        except Exception as e:
            st.error(f"Could not load checkpoint: {e}")

# Evaluate
if st.button("Evaluate on Test Set"):
    acc, bleu, preds = evaluate(st.session_state.model, test_X, test_Y)
    st.metric("Test Accuracy", f"{acc*100:.1f}%")
    st.metric("Test BLEU-1", f"{bleu*100:.1f}%")

st.markdown("---")
st.subheader("Try a custom sequence")
user_seq = st.text_input("Enter 10 integers 1–19 (space-separated)", "1 2 3 4 5 6 7 8 9 10")

if st.button("Predict & Visualize"):
    try:
        toks = [int(x) for x in user_seq.strip().split()]
        assert len(toks) == 10 and all(1 <= t <= 19 for t in toks)
    except:
        st.error("Please enter exactly 10 integers in [1..19].")
        st.stop()

    src = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(DEVICE)   # [1, S]
    # teacher forcing for visualization (target input is reversed but shifted)
    tgt_ref = torch.flip(src, dims=[1])                                  # [1, T]
    tgt_inp = tgt_ref[:, :-1]                                            # shift for teacher forcing
    mask = causal_mask(tgt_inp.size(1))

    with torch.no_grad():
        logits = st.session_state.model(src, tgt_inp, tgt_mask=mask)
        pred_ids = logits.argmax(-1).squeeze(0).tolist()

    st.write("**Input** :", toks)
    st.write("**Ref**   :", tgt_ref.squeeze(0).tolist()[1:])   # ref aligned with outputs
    st.write("**Pred**  :", pred_ids)

    # Attention
    W = get_cross_attn_weights_trained(st.session_state.model, src, tgt_inp, tgt_mask=mask)
    plot_attn(W, title="Decoder Cross-Attention (trained layer)")


st.caption("Tip: expect an anti-diagonal attention for the reverse task.")
