# train.py  —  IMDB BiLSTM + Attention with tiny focus regularizers (CPU-friendly)
# Saves: bilstm.pt, vocab.json, config.json

import json, random, time, re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

# ---------------------------
# Reproducibility & env
# ---------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"[INFO] Torch {torch.__version__} | Device=CPU")

# ---------------------------
# Lightweight CPU config
# (Adjust subset_* or epochs if you need even faster)
# ---------------------------
OUT_DIR = Path("."); OUT_DIR.mkdir(exist_ok=True)
CONFIG = {
    "embed_dim": 64,
    "hidden_dim": 64,
    "num_classes": 2,
    "batch_size": 64,
    "epochs": 3,            # conservative; raise to 4–5 if you want
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "max_vocab": 15000,
    "min_freq": 2,
    "max_len": 256,         # shorter seqs -> faster on CPU
    "subset_train": 3000,   # was 1000 in your fast script
    "subset_valid": 800,
    "subset_test": 800,
    "num_workers": 0        # keep 0 on Windows; 2 is fine on Linux/mac
}

print("[CONFIG]")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# ---------------------------
# Tokenizer / Vocab
# ---------------------------
_word_re = re.compile(r"[A-Za-z0-9']+")
def tokenize(text: str):
    return _word_re.findall(text.lower())

def build_vocab(texts, max_vocab=20000, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    items = [(tok, c) for tok, c in counter.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[:max_vocab - 2]  # reserve <pad>=0, <unk>=1
    stoi = {"<pad>": 0, "<unk>": 1}
    for i, (tok, _) in enumerate(items, start=2):
        stoi[tok] = i
    return stoi

def numericalize(text, stoi, max_len=400):
    ids = [stoi.get(tok, 1) for tok in tokenize(text)]
    if not ids: ids = [1]
    if len(ids) > max_len: ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)

# ---------------------------
# Dataset
# ---------------------------
class TextDS(Dataset):
    def __init__(self, samples, stoi, max_len=400):
        self.samples = samples
        self.stoi = stoi
        self.max_len = max_len
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        x = numericalize(s["text"], self.stoi, self.max_len)  # tensor
        y = int(s["label"])
        return x, len(x), y

def collate(batch):
    xs, lens, ys = zip(*batch)
    padded = pad_sequence(xs, batch_first=True, padding_value=0)
    lengths = torch.tensor(lens, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return padded, lengths, y

# ---------------------------
# Model: BiLSTM + additive attention pooling
# ---------------------------
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
        emb = self.embedding(x)                           # [B,T,E]
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)  # [B,T,2H]

        max_T = out.size(1)
        mask = (torch.arange(max_T, device=lengths.device)[None, :] < lengths[:, None]).float()  # [B,T]

        scores = self.attn_v(torch.tanh(self.attn_w(out))).squeeze(-1)   # [B,T]
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=1)                               # [B,T]

        ctx = torch.bmm(attn.unsqueeze(1), out).squeeze(1)                # [B,2H]
        logits = self.fc(ctx)                                             # [B,C]
        if return_attn:
            return logits, attn, mask
        return logits

# ---------------------------
# Data via HF Datasets
# ---------------------------
def take_subset(dataset, n):
    idx = list(range(len(dataset)))
    random.shuffle(idx)
    idx = idx[:n]
    return [dataset[i] for i in idx]

def load_imdb_subsets():
    print("[INFO] Loading IMDB from Hugging Face…")
    ds = load_dataset("imdb")
    train_full, test_full = ds["train"], ds["test"]
    print("[INFO] Building vocab on full train set (one pass)…")
    stoi = build_vocab((ex["text"] for ex in train_full),
                       max_vocab=CONFIG["max_vocab"], min_freq=CONFIG["min_freq"])
    print(f"[INFO] Vocab size: {len(stoi)}")
    print("[INFO] Sampling CPU-friendly subsets…")
    train_samples = take_subset(train_full, CONFIG["subset_train"])
    valid_samples = take_subset(train_full, CONFIG["subset_valid"])
    test_samples  = take_subset(test_full,  CONFIG["subset_test"])
    return stoi, train_samples, valid_samples, test_samples

# ---------------------------
# Train / Eval
# ---------------------------
STOPWORDS = {
    "the","a","an","and","or","but","if","in","on","at","for","to","of","is","are","was","were","be","been",
    "this","that","it","its","i","you","he","she","they","we","me","him","her","them","my","your","their",
    "with","as","by","from","about","so","just","very","really","there","here","then","than","also","too",
    "movie","film"
}
stop_ids_vec = None  # set after vocab is built

def run_epoch(model, loader, crit, opt=None, desc="train"):
    training = opt is not None
    model.train(training)
    losses, preds_all, ys_all = [], [], []
    pbar = tqdm(loader, desc=f"{desc.upper()}  ", leave=False)
    for xb, lengths, yb in pbar:
        if training:
            logits, attn, mask = model(xb, lengths, return_attn=True)
            loss = crit(logits, yb)
            # --- tiny attention regularizers ---
            ent = -(attn * (attn.clamp_min(1e-8).log())).sum(dim=1).mean()
            stop_mask = stop_ids_vec.to(xb.device)[xb] * mask
            attn_stop = (attn * stop_mask).sum(dim=1).mean()
            loss = loss + 0.001 * ent + 0.01 * attn_stop
            # backprop
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step()
        else:
            logits = model(xb, lengths, return_attn=False)
            loss = crit(logits, yb)

        losses.append(loss.item())
        preds_all.append(logits.argmax(1).detach().cpu())
        ys_all.append(yb.detach().cpu())
        if len(losses) % 10 == 0:
            pbar.set_postfix(loss=f"{np.mean(losses[-10:]):.3f}")

    y_true = torch.cat(ys_all).numpy()
    y_pred = torch.cat(preds_all).numpy()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return float(np.mean(losses)), acc, f1

def main():
    t0_all = time.time()
    global stop_ids_vec

    stoi, train_s, valid_s, test_s = load_imdb_subsets()
    vocab_size = max(stoi.values()) + 1
    print(f"[INFO] Sizes — train={len(train_s)} valid={len(valid_s)} test={len(test_s)}")

    # build stopword id vector once
    stop_ids_vec = torch.zeros(vocab_size, dtype=torch.float32)
    for w in STOPWORDS:
        if w in stoi: stop_ids_vec[stoi[w]] = 1.0

    train_ds = TextDS(train_s, stoi, CONFIG["max_len"])
    valid_ds = TextDS(valid_s, stoi, CONFIG["max_len"])
    test_ds  = TextDS(test_s,  stoi, CONFIG["max_len"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                              collate_fn=collate, num_workers=CONFIG["num_workers"])
    valid_loader = DataLoader(valid_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                              collate_fn=collate, num_workers=CONFIG["num_workers"])
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False,
                              collate_fn=collate, num_workers=CONFIG["num_workers"])

    model = BiLSTMWithAttn(vocab_size, CONFIG["embed_dim"], CONFIG["hidden_dim"], CONFIG["num_classes"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    best_f1, best_state = -1.0, None
    print("\n[TRAINING]")
    for epoch in range(1, CONFIG["epochs"] + 1):
        t_epoch = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, desc=f"epoch {epoch} train")
        va_loss, va_acc, va_f1 = run_epoch(model, valid_loader, criterion, opt=None, desc=f"epoch {epoch} valid")
        dt = time.time() - t_epoch
        print(f"Epoch {epoch}/{CONFIG['epochs']} | {dt:.1f}s | "
              f"Train: loss={tr_loss:.3f} acc={tr_acc:.3f} f1={tr_f1:.3f} | "
              f"Val: loss={va_loss:.3f} acc={va_acc:.3f} f1={va_f1:.3f}")
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print("  ↳ [BEST] New best val F1; checkpointed in-memory.")

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n[TEST]")
    te_loss, te_acc, te_f1 = run_epoch(model, test_loader, criterion, opt=None, desc="test")
    print(f"Test: loss={te_loss:.3f} acc={te_acc:.3f} f1={te_f1:.3f}")

    # Save artifacts
    torch.save(model.state_dict(), OUT_DIR / "bilstm.pt")
    with open(OUT_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(stoi, f)
    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump({"pad_idx": 0, **CONFIG, "vocab_size": vocab_size}, f, indent=2)

    print(f"\n[SAVED] bilstm.pt, vocab.json, config.json  |  Total time: {time.time() - t0_all:.1f}s")

if __name__ == "__main__":
    main()
