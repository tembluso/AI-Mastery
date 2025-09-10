import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="Fashion-MNIST Mini-Project", layout="wide")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# ---------------- Data ----------------
@st.cache_resource(show_spinner=True)
def load_data(batch_size=128):
    tfm = transforms.ToTensor()
    train_full = datasets.FashionMNIST(root="data", train=True, download=True, transform=tfm)
    train_ds, val_ds = random_split(train_full, [50000, 10000])
    test_ds = datasets.FashionMNIST(root="data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1000)
    test_loader  = DataLoader(test_ds, batch_size=1000)
    return train_loader, val_loader, test_loader, test_ds

# ---------------- Model ----------------
class FashionNet(nn.Module):
    def __init__(self, hidden1=512, hidden2=256, hidden3=128, dropout=0.4):
        super().__init__()
        self.fc1 = nn.Linear(28*28, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 10)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        x = self.drop(self.relu(self.fc3(x)))
        return self.fc4(x)

def make_optimizer(params, name, lr, weight_decay):
    if name == "Adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

@st.cache_resource(show_spinner=False)
def init_model(h1, h2, h3, dropout):
    model = FashionNet(h1, h2, h3, dropout).to(DEVICE)
    return model

def evaluate(model, loader, crit):
    model.eval()
    total_loss=0; correct=0; total=0
    with torch.no_grad():
        for X,y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            total_loss += crit(logits, y).item()
            preds = logits.argmax(1)
            correct += (preds==y).sum().item()
            total += y.size(0)
    return total_loss/len(loader), correct/total

def train(model, train_loader, val_loader, optimizer, epochs=5):
    crit = nn.CrossEntropyLoss()
    train_losses=[]; val_losses=[]; val_accs=[]
    prog = st.progress(0)
    status = st.empty()
    for ep in range(epochs):
        model.train(); running=0.0
        for X,y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = crit(logits, y)
            loss.backward(); optimizer.step()
            running += loss.item()
        train_losses.append(running/len(train_loader))
        vloss, vacc = evaluate(model, val_loader, crit)
        val_losses.append(vloss); val_accs.append(vacc)
        prog.progress(int((ep+1)/epochs*100))
        status.write(f"Epoch {ep+1}/{epochs} — train loss {train_losses[-1]:.4f} | val loss {vloss:.4f} | val acc {vacc:.3f}")
    return train_losses, val_losses, val_accs

def predict_topk(model, X, k=3):
    model.eval()
    with torch.no_grad():
        logits = model(X.to(DEVICE))
        probs = torch.softmax(logits, dim=1)
        top_probs, top_idx = probs.topk(k, dim=1)
    return top_idx.cpu().numpy(), top_probs.cpu().numpy()

# ---------------- UI ----------------
st.title("👕 Fashion-MNIST — Week 6 Mini-Project")
st.write("Train a model, watch live metrics, browse predictions and misclassifications — all in one app.")

with st.sidebar:
    st.header("Configuration")
    batch_size = st.slider("Batch size", 64, 256, 128, step=32)
    epochs = st.slider("Epochs", 1, 20, 8)
    optimizer_name = st.selectbox("Optimizer", ["Adam","SGD+Momentum"])
    lr = st.select_slider("Learning rate", options=[1e-3, 5e-4, 1e-4], value=1e-3, format_func=lambda x: f"{x:.0e}")
    weight_decay = st.selectbox("Weight decay", [0.0, 1e-4, 1e-3], index=1, format_func=lambda x: f"{x:.0e}")
    dropout = st.slider("Dropout", 0.0, 0.7, 0.4, 0.1)
    h1 = st.selectbox("Hidden1", [256,512,768], index=1)
    h2 = st.selectbox("Hidden2", [128,256,384], index=1)
    h3 = st.selectbox("Hidden3", [64,128,192], index=1)
    st.markdown("---")
    go = st.button("🚀 Train / Retrain")

# Load data
train_loader, val_loader, test_loader, test_ds = load_data(batch_size=batch_size)
model = init_model(h1,h2,h3,dropout)
optimizer = make_optimizer(model.parameters(), optimizer_name, lr, weight_decay)

tab1, tab2, tab3 = st.tabs(["📈 Training", "🔎 Predictions", "📊 Confusion Matrix"])

with tab1:
    if go:
        tr, vl, vacc = train(model, train_loader, val_loader, optimizer, epochs=epochs)
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].plot(tr, label="Train Loss"); ax[0].plot(vl, label="Val Loss"); ax[0].legend(); ax[0].set_title("Loss")
        ax[1].plot(vacc, label="Val Acc"); ax[1].legend(); ax[1].set_title("Validation Accuracy")
        st.pyplot(fig)
        # Cache weights to disk
        torch.save(model.state_dict(), "fashion_net.pt")
        st.success("Saved weights to fashion_net.pt")
    else:
        st.info("Set your config on the left and click **Train / Retrain**.")

with tab2:
    st.subheader("Browse Predictions")
    if Path("fashion_net.pt").exists():
        model.load_state_dict(torch.load("fashion_net.pt", map_location=DEVICE))
        model.eval()
        st.success("Loaded saved weights.")
    else:
        st.warning("No saved weights found yet. Train the model first.")

    import random
    n_samples = st.slider("Number of samples", 5, 30, 12)
    only_errors = st.checkbox("Show only misclassifications", False)

    idxs = random.sample(range(len(test_ds)), 50)
    images=[]; labels=[]; preds=[]; top3=[]
    with torch.no_grad():
        for i in idxs:
            x,y = test_ds[i]
            images.append(x.squeeze(0).numpy())
            labels.append(y)
        X = torch.stack([torch.tensor(img) for img in images]).unsqueeze(1).float().to(DEVICE)
        logits = model(X)
        p = logits.argmax(1).cpu().numpy()
        preds = p.tolist()
        t_idx, t_probs = predict_topk(model, X, k=3)
        top3 = [(idx, probs) for idx, probs in zip(t_idx, t_probs)]
    data = list(zip(images, labels, preds, top3))
    if only_errors:
        data = [d for d in data if d[1] != d[2]]
    data = data[:n_samples]

    cols = st.columns(4)
    for i, (img, y, yhat, (tidx, tprob)) in enumerate(data):
        with cols[i%4]:
            st.image(img, width=120, caption=f"true: {CLASSES[y]}\npred: {CLASSES[yhat]}")
            st.write("Top-3: " + ", ".join([f"{CLASSES[int(tidx[j])]} {float(tprob[j]):.2f}" for j in range(3)]))

with tab3:
    st.subheader("Confusion Matrix (Test)")
    if Path("fashion_net.pt").exists():
        model.load_state_dict(torch.load("fashion_net.pt", map_location=DEVICE))
        model.eval()
        y_true=[]; y_pred=[]
        with torch.no_grad():
            for X,y in test_loader:
                X = X.to(DEVICE)
                logits = model(X)
                y_true.extend(y.numpy())
                y_pred.extend(logits.argmax(1).cpu().numpy())
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        st.caption("Tip: Look for off-diagonal blocks (e.g., Shirt vs T-shirt).")
    else:
        st.info("Train the model first to generate predictions for the confusion matrix.")
