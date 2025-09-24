# train.py — trains TinyTransformer on reverse task and saves checkpoint.pt

import numpy as np
np.random.seed(42)
import torch, torch.nn as nn, torch.optim as optim
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_data(num_samples=2000, seq_len=10, vocab_size=20):
    X, Y = [], []
    for _ in range(num_samples):
        seq = torch.randint(1, vocab_size, (seq_len,))
        X.append(seq); Y.append(torch.flip(seq, dims=[0]))
    return torch.stack(X), torch.stack(Y)

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=128, num_layers=2, max_len=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.enc = nn.ModuleList([nn.TransformerEncoderLayer(d_model, num_heads, d_ff, batch_first=True) for _ in range(num_layers)])
        self.dec = nn.ModuleList([nn.TransformerDecoderLayer(d_model, num_heads, d_ff, batch_first=True) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, src, tgt, tgt_mask=None):
        src_e = self.src_emb(src) + self.pe[:, :src.size(1)]
        tgt_e = self.tgt_emb(tgt) + self.pe[:, :tgt.size(1)]
        mem = src_e
        for l in self.enc: mem = l(mem)
        out = tgt_e
        for l in self.dec: out = l(out, mem, tgt_mask=tgt_mask)
        return self.fc(out)

def causal_mask(T): 
    m = torch.triu(torch.ones(T, T, device=DEVICE), diagonal=1)
    return m.masked_fill(m==1, float("-inf"))

def main():
    X, Y = make_data(2000, 10, 20)
    split = 1600
    train_X, train_Y = X[:split], Y[:split]
    test_X,  test_Y  = X[split:],  Y[split:]

    model = TinyTransformer(20).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(train_X, train_Y)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)

    for ep in range(40):
        model.train(); tot=0.0
        for Xb, Yb in dl:
            Xb = Xb.to(DEVICE)
            tgt_inp = Yb[:, :-1].to(DEVICE)
            tgt_out = Yb[:, 1:].to(DEVICE)
            mask = causal_mask(tgt_inp.size(1))
            logits = model(Xb, tgt_inp, tgt_mask=mask)
            loss = crit(logits.reshape(-1, 20), tgt_out.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"Epoch {ep+1:02d}: loss {tot/len(dl):.4f}")

    torch.save(model.state_dict(), "checkpoint.pt")
    print("Saved checkpoint.pt")

if __name__ == "__main__":
    main()
