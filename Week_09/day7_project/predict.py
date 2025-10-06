# predict.py
# CLI + helper that loads the single training bundle (bundle.pt).

import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchtext.data.utils import get_tokenizer
from typing import Dict, Tuple

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=2, pad_idx=0, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(self.dropout(h[-1]))

def load_bundle(bundle_path: str = "bundle.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torch.load(bundle_path, map_location=device)
    vocab: Dict[str, int] = bundle["vocab"]
    pad_idx = vocab.get(bundle.get("pad_token", "<PAD>"), 0)
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=bundle.get("embed_dim", 64),
        hidden_dim=bundle.get("hidden_dim", 128),
        num_classes=bundle.get("num_classes", 2),
        pad_idx=pad_idx,
    ).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    tok = get_tokenizer(bundle.get("tokenizer", "basic_english"))
    unk_idx = vocab.get(bundle.get("unk_token", "<UNK>"), 1)
    max_len = bundle.get("max_len", None)
    return model, tok, vocab, device, pad_idx, unk_idx, max_len

@torch.no_grad()
def predict_text(text: str, model, tok, vocab, device, pad_idx, unk_idx, max_len=None, return_probs=False):
    ids = [vocab.get(t, unk_idx) for t in tok(text)]
    if max_len is not None:
        ids = ids[:max_len]
    if not ids:
        ids = [unk_idx]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    L = torch.tensor([len(ids)], dtype=torch.long, device=device)
    logits = model(x, L)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    idx = int(torch.tensor(probs).argmax())
    label = "Positive" if idx == 0 else "Negative"
    out = {"label": label, "prob": float(probs[idx])}
    if return_probs:
        out["probs"] = {"pos": float(probs[0]), "neg": float(probs[1])}
    return out

if __name__ == "__main__":
    import sys
    model, tok, vocab, device, pad_idx, unk_idx, max_len = load_bundle("bundle.pt")
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "This movie was fantastic, I loved it!"
    res = predict_text(text, model, tok, vocab, device, pad_idx, unk_idx, max_len, return_probs=True)
    print(res)
