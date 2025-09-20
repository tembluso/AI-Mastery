# app.py — Streamlit app with Integrated Gradients saliency

import json, re
from typing import List, Tuple
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# ---------- tokenizer ----------
_word_re = re.compile(r"[A-Za-z0-9']+")
def simple_tokenize(text: str) -> List[str]:
    return _word_re.findall(text.lower())

# ---------- model (mirror training) ----------
class BiLSTMWithAttn(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64,
                 num_classes=2, pad_idx=0, dropout=0.3, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=True, dropout=dropout, num_layers=num_layers)
        self.attn_w = nn.Linear(hidden_dim*2, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x, lengths, return_attn=False):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        max_T = out.size(1)
        mask = (torch.arange(max_T, device=lengths.device)[None, :] < lengths[:, None]).float()
        scores = self.attn_v(torch.tanh(self.attn_w(out))).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=1)
        ctx = torch.bmm(attn.unsqueeze(1), out).squeeze(1)
        logits = self.fc(ctx)
        if return_attn:
            return logits, attn, mask
        return logits

@st.cache_resource
def load_artifacts():
    with open("vocab.json", "r", encoding="utf-8") as f:
        stoi = json.load(f)
    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    vocab_size = max(stoi.values()) + 1
    model = BiLSTMWithAttn(
        vocab_size=vocab_size,
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=cfg["num_classes"],
        pad_idx=0,
        dropout=0.3,
        num_layers=2
    )
    state = torch.load("bilstm.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, stoi, cfg

def ids_from_text(text: str, stoi, max_len: int):
    toks = simple_tokenize(text)
    ids = [stoi.get(tok, 1) for tok in toks]
    if len(ids) == 0: ids = [1]; toks = ["<unk>"]
    if len(ids) > max_len: ids = ids[:max_len]; toks = toks[:max_len]
    return torch.tensor(ids, dtype=torch.long), toks

# ---------- saliency: Integrated Gradients ----------
@torch.inference_mode(False)
def saliency_for_text(model, ids_1d: torch.Tensor, steps: int = 32):
    T = ids_1d.numel()
    x = ids_1d.unsqueeze(0)
    lengths = torch.tensor([T], dtype=torch.long)

    with torch.no_grad():
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        label_id = int(np.argmax(probs))

    base = torch.zeros(1, T, model.embedding.embedding_dim)
    total = torch.zeros_like(base)
    for alpha in torch.linspace(0.0, 1.0, steps):
        emb = model.embedding(x).detach()
        emb = base + alpha * (emb - base)
        emb.requires_grad_(True)

        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = model.lstm(packed)
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        logits = model.fc(h_cat)
        loss = logits[0, label_id]

        model.zero_grad(set_to_none=True)
        loss.backward()
        total = total + emb.grad

    inp = model.embedding(x).detach()
    ig = (inp - base) * (total / steps)
    scores = ig.norm(dim=2)[0].detach().cpu().numpy()
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.zeros_like(scores)
    return probs, label_id, scores

def colorize_tokens_html(tokens: List[str], scores: np.ndarray, clip_low: float = 0.05) -> str:
    html_parts = []
    for tok, s in zip(tokens, scores):
        a = float(max(0.0, s - clip_low) / max(1e-6, 1.0 - clip_low))
        bg = f"rgba(255, 0, 0, {a*0.55:.3f})"
        html_parts.append(f"<span style='background:{bg}; padding:2px 3px; border-radius:4px; margin:1px'>{tok}</span>")
    return " ".join(html_parts)

# ---------- UI ----------
st.set_page_config(page_title="IMDB Sentiment (BiLSTM+Attn) + IG", page_icon="🎬", layout="centered")
st.title("🎬 IMDB Sentiment — Focused Highlights")
st.caption("Predicts sentiment and highlights influential tokens (Integrated Gradients).")

model, stoi, cfg = load_artifacts()

with st.form("inference_form"):
    text_block = st.text_area(
        "Enter one or many reviews (one per line):",
        height=180,
        placeholder="This movie was surprisingly heartfelt and well-acted.\nI hated the pacing and the acting felt flat."
    )
    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.form_submit_button("Analyze")
    with col2:
        show_probs = st.checkbox("Show class probability bars", value=True)

if run_btn:
    reviews = [ln.strip() for ln in text_block.splitlines() if ln.strip()]
    if not reviews:
        st.warning("Please enter at least one review.")
    else:
        st.write(f"**Processing {len(reviews)} review(s)…**")
        for i, review in enumerate(reviews, start=1):
            ids, toks = ids_from_text(review, stoi, cfg["max_len"])
            probs, label_id, scores = saliency_for_text(model, ids)
            label = "Positive ✅" if label_id == 1 else "Negative ❌"
            st.markdown(f"### {i}. {label}")
            if show_probs:
                st.write(f"**neg:** {probs[0]:.3f}  |  **pos:** {probs[1]:.3f}")
                st.progress(float(probs[label_id]))
            html = colorize_tokens_html(toks, scores)
            st.markdown(html, unsafe_allow_html=True)
            st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Train: `python train.py`  →  Run: `streamlit run app.py`")
