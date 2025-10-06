# train.py  — IMDB LSTM (simple, with robust label handling + progress prints)

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# ---------------------- Settings (edit here if needed) ----------------------
EPOCHS      = 2
BATCH_SIZE  = 32
EMBED_DIM   = 64
HIDDEN_DIM  = 128
VOCAB_SIZE  = 20000
MAX_LEN     = 200     # truncate long reviews for speed
LR          = 1e-3
CLIP_NORM   = 5.0

PAD, UNK = "<PAD>", "<UNK>"

print("Step 1: Device & seed")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ---------------------- Helpers ----------------------
def normalize_label(label):
    """
    Robust mapping for various IMDB label formats across torchtext builds.
    Maps:
      'pos', 'positive', '1' -> 1 (positive)
      'neg', 'negative', '0', '2' -> 0 (negative)
    """
    s = str(label).strip().lower()

    # explicit numeric cases first (your build shows '1' and '2')
    if s == "1":  # positive
        return 1
    if s == "2":  # negative
        return 0
    if s == "0":  # some corpora use 0 for negative
        return 0

    # common text labels
    if s.startswith("pos") or s in {"positive", "pos/1", "pos\t"}:
        return 1
    if s.startswith("neg") or s in {"negative", "neg/0", "neg\t"}:
        return 0

    # last-resort heuristics
    if "pos" in s: return 1
    if "neg" in s: return 0

    raise ValueError(f"Unknown IMDB label format: {label!r}")


def encode(text, tokenizer, vocab):
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokenizer(text)]
    if MAX_LEN is not None:
        ids = ids[:MAX_LEN]
    return torch.tensor(ids, dtype=torch.long)

# ---------------------- Load data ----------------------
print("Step 2: Load IMDB splits")
train_iter = list(IMDB(split="train"))
test_iter  = list(IMDB(split="test"))

print("Step 2.1: Sanity-check labels")
try:
    train_counts = Counter(normalize_label(l) for l, _ in train_iter)
    test_counts  = Counter(normalize_label(l) for l, _ in test_iter)
    print("  Train label counts:", train_counts, "(0=neg, 1=pos)")
    print("  Test  label counts:", test_counts,  "(0=neg, 1=pos)")
    print("  Sample raw labels:", [train_iter[i][0] for i in range(3)])
except Exception as e:
    unique = sorted({str(l).strip().lower() for l, _ in train_iter})
    print("  ERROR normalizing labels:", e)
    print("  Unique raw labels seen in train:", unique)
    raise

# ---------------------- Build vocab ----------------------
print("Step 3: Build vocab")
tokenizer = get_tokenizer("basic_english")
counter = Counter()
for label, line in train_iter:
    counter.update(tokenizer(line))

vocab = {PAD: 0, UNK: 1}
for i, (w, _) in enumerate(counter.most_common(VOCAB_SIZE - 2), start=2):
    vocab[w] = i

def collate_fn(batch):
    seqs, labels, lens = [], [], []
    for label, text in batch:
        seq = encode(text, tokenizer, vocab)
        seqs.append(seq)
        labels.append(normalize_label(label))   # << robust label mapping
        lens.append(len(seq))
    seqs = pad_sequence(seqs, batch_first=True, padding_value=vocab[PAD])
    return seqs, torch.tensor(lens), torch.tensor(labels)

print("Step 4: DataLoaders")
train_loader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
test_loader  = DataLoader(test_iter,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ---------------------- Model ----------------------
print("Step 5: Define model")
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(self.dropout(h[-1]))

model = LSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, vocab[PAD]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------------------- Eval ----------------------
@torch.no_grad()
def evaluate():
    model.eval()
    all_y, all_p = [], []
    for X, L, y in test_loader:
        X, L, y = X.to(device), L.to(device), y.to(device)
        logits = model(X, L)
        pred = logits.argmax(1)
        all_y.extend(y.tolist())
        all_p.extend(pred.tolist())

    # Metrics (force both classes so sklearn never crashes)
    labels = [0, 1]
    acc = accuracy_score(all_y, all_p)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_y, all_p, average="binary", pos_label=1, zero_division=0
    )

    print("\n--- Classification Report ---")
    print(classification_report(
        all_y, all_p, labels=labels, target_names=["neg", "pos"], zero_division=0
    ))
    print("Counts | y_true:", Counter(all_y), "| y_pred:", Counter(all_p))
    return acc, prec, rec, f1

# ---------------------- Train ----------------------
print("Step 6: Train")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for i, (X, L, y) in enumerate(train_loader, 1):
        X, L, y = X.to(device), L.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X, L)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        running += loss.item()
        if i % 100 == 0:
            print(f"  Batch {i}/{len(train_loader)} | Loss={loss.item():.4f}")
    train_loss = running / max(1, len(train_loader))
    acc, prec, rec, f1 = evaluate()
    print(f"Epoch {epoch}: TrainLoss={train_loss:.4f} | Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f}")

# ---------------------- Save ----------------------
print("Step 7: Save bundle")
torch.save({
    "state_dict": model.state_dict(),
    "vocab": vocab,
    "pad_token": PAD,
    "unk_token": UNK,
    "embed_dim": EMBED_DIM,
    "hidden_dim": HIDDEN_DIM,
    "num_classes": 2,
    "tokenizer": "basic_english",
    "max_len": MAX_LEN,
}, "bundle.pt")
print("Done → bundle.pt")
