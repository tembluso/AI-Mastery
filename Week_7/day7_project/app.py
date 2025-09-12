import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path

# -------------------------
# App Config
# -------------------------
st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🤖", layout="centered")
st.title("CIFAR-10 Mini App — Baseline CNN")
st.caption("Week 7 · Day 7 — Upload an image, get top‑3 predictions.")

# -------------------------
# Model Definition (must match training)
# -------------------------
class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# -------------------------
# Utilities
# -------------------------
CLASSES = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str = "../cifar10_baseline_best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR_CNN().to(device)
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        st.warning("Checkpoint not found. Please place **cifar_cnn_best.pt** alongside app.py.")
        return model.eval(), device, False
    state = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device, True

TRANSFORM = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

@torch.inference_mode()
def predict_topk(model, device, pil_img: Image.Image, k: int = 3):
    x = TRANSFORM(pil_img.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    topk_idx = probs.argsort()[-k:][::-1]
    return [(CLASSES[i], float(probs[i])) for i in topk_idx]

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Settings")
ckpt = st.sidebar.text_input("Checkpoint path", value="../cifar10_baseline_best.pt")
K = st.sidebar.slider("Top‑K", min_value=1, max_value=5, value=3, step=1)

model, device, has_ckpt = load_model(ckpt)

# -------------------------
# Main UI
# -------------------------
uploaded = st.file_uploader("Upload an image (any size); we'll center‑crop/resize to 32×32", type=["png","jpg","jpeg","webp"])

col1, col2 = st.columns([1,1])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    col1.subheader("Input Image")
    col1.image(img, width="stretch")

    if not has_ckpt:
        st.stop()

    with st.spinner("Running inference..."):
        preds = predict_topk(model, device, img, k=K)

    # Results table
    labels = [p[0] for p in preds]
    scores = [p[1] for p in preds]

    col2.subheader("Top Predictions")
    col2.write(
        "\n".join([f"**{lbl}** — {score*100:.1f}%" for lbl, score in preds])
    )

    # Simple bar chart
    st.subheader("Probability Chart")
    chart_data = {"class": labels, "prob": scores}
    st.bar_chart(chart_data, x="class", y="prob")

else:
    st.info("👆 Upload an image to see predictions.")

# Footer
st.caption(
    "Model: 3‑block CNN trained on CIFAR‑10 with augmentation. "
    "Place **cifar_cnn_best.pt** next to this file. Run: `streamlit run app.py`"
)
