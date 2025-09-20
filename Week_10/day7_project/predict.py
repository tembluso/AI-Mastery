# predict.py — Batch predictions + IG saliency (matches train/app)

import json, re
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# ---------- tokenizer ----------
_word_re = re.compile(r"[A-Za-z0-9']+")
def simple_tokenize(text: str) -> List[str]:
    return _word_re.findall(text.lower())

def ids_from_text(text: str, stoi: Dict[str, int], max_len: int):
    toks = simple_tokenize(text)
    ids = [stoi.get(tok, 1) for tok in toks]
    if len(ids) == 0: ids = [1]; toks = ["<unk>"]
    if len(ids) > max_len: ids = ids[:max_len]; toks = toks[:max_len]
    x = torch.tensor(ids, dtype=torch.long)
    return x, toks

# ---------- model ----------
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

def load_artifacts(ckpt="bilstm.pt", vocab_path="vocab.json", cfg_path="config.json"):
    with open(vocab_path, "r", encoding="utf-8") as f:
        stoi = json.load(f)
    with open(cfg_path, "r", encoding="utf-8") as f:
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
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, stoi, cfg

# ---------- Integrated Gradients saliency (single sample) ----------
@torch.inference_mode(False)
def saliency_ig(model: BiLSTMWithAttn, ids_1d: torch.Tensor, target_class: int = None, steps: int = 32):
    T = ids_1d.numel()
    x = ids_1d.unsqueeze(0)
    lengths = torch.tensor([T], dtype=torch.long)

    with torch.no_grad():
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(probs))
    cls = pred_class if target_class is None else int(target_class)

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
        loss = logits[0, cls]

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
    return probs, pred_class, scores

# ---------- public APIs ----------
def predict_one(text: str, model=None, stoi=None, cfg=None, saliency=True):
    if model is None:
        model, stoi, cfg = load_artifacts()
    ids, toks = ids_from_text(text, stoi, cfg["max_len"])
    lengths = torch.tensor([ids.numel()], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids.unsqueeze(0), lengths)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    label_id = int(np.argmax(probs))
    out = {
        "label": "pos" if label_id == 1 else "neg",
        "probs": {"neg": float(probs[0]), "pos": float(probs[1])},
        "tokens": toks
    }
    if saliency:
        _, _, scores = saliency_ig(model, ids, target_class=label_id, steps=32)
        out["saliency"] = scores.tolist()
    return out

def predict_batch(texts: List[str], model=None, stoi=None, cfg=None, saliency=True) -> List[Dict[str, Any]]:
    if model is None:
        model, stoi, cfg = load_artifacts()
    return [predict_one(t, model, stoi, cfg, saliency=saliency) for t in texts]

# -------------- CLI --------------
if __name__ == "__main__":
    import argparse, json as _json
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+", help="One or more texts (quote each)")
    ap.add_argument("--nosal", action="store_true", help="Disable saliency")
    args = ap.parse_args()
    outs = predict_batch(args.text, saliency=not args.nosal)
    print(_json.dumps(outs, indent=2))
