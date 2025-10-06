# app.py — Streamlit Transfer-Learning Demo with Grad-CAM
import streamlit as st
import torch, torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np

# -----------------------------
# Config
# -----------------------------
CLASSES = ["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]  # replace with your labels
CKPT = "../best_resnet18.pt"

st.set_page_config(page_title="Transfer Learning Demo", page_icon="🖼️", layout="centered")
st.title("Week 8 · Day 7 — Transfer Learning Demo")
st.caption("Upload an image → see top-3 predictions + Grad-CAM heatmap.")

# -----------------------------
# Model
# -----------------------------
@st.cache_resource
def load_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_feats = m.fc.in_features
    m.fc = nn.Linear(in_feats, len(CLASSES))
    m.load_state_dict(torch.load(CKPT, map_location="cpu"))
    m.eval()
    return m

model = load_model()
target_layer = model.layer4[1].conv2

# -----------------------------
# Preprocessing
# -----------------------------
TRANSFORM = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

def gradcam(model, img_tensor, target_layer, class_idx=None):
    feats = None; grads = None
    def fwd_hook(_,__,out): nonlocal feats; feats = out.detach()
    def bwd_hook(_,gin,gout): nonlocal grads; grads = gout[0].detach()
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_backward_hook(bwd_hook)
    out = model(img_tensor)
    if class_idx is None: class_idx = int(out.argmax(1))
    score = out[0,class_idx]; model.zero_grad(); score.backward()
    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = (weights*feats).sum(dim=1).relu()[0].cpu().numpy()
    cam = (cam-cam.min())/(cam.max()+1e-8)
    h1.remove(); h2.remove()
    return cam, class_idx, torch.softmax(out,dim=1)[0].detach().cpu().numpy()

# -----------------------------
# UI
# -----------------------------
img_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp"])
if img_file:
    pil_img = Image.open(img_file).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", width="stretch")
    img_tensor = TRANSFORM(pil_img).unsqueeze(0)
    cam, pred_idx, probs = gradcam(model, img_tensor, target_layer)
    # Top-3
    top3 = np.argsort(probs)[-3:][::-1]
    st.subheader("Predictions")
    for i in top3:
        st.write(f"**{CLASSES[i]}** — {probs[i]*100:.1f}%")
    # Grad-CAM overlay
    import matplotlib.pyplot as plt
    import io
    disp = np.array(pil_img.resize((224,224)))/255.0
    plt.imshow(disp); plt.imshow(cam, cmap="jet", alpha=0.45)
    plt.axis("off"); buf = io.BytesIO(); plt.savefig(buf, format="png"); plt.close()
    st.image(buf, caption=f"Grad-CAM (True heatmap for Pred: {CLASSES[pred_idx]})")
else:
    st.info("👆 Upload an image to get predictions and Grad-CAM.")
