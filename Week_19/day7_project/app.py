# app.py
# Streamlit demo: Efficiency Showdown (CPU-only)
# Baseline (Teacher) vs Quantized (Teacher int8) vs Student (Distilled)

import os, time, copy
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import torch, torch.nn as nn
from torchvision import datasets, transforms

# ----- Setup -----
torch.set_num_threads(4)
device = torch.device("cpu")
ART = Path("artifacts"); ART.mkdir(exist_ok=True)

st.set_page_config(page_title="Efficiency Showdown (CPU-only)", layout="wide")
st.title("📊 Efficiency Showdown — Baseline vs Quantized vs Distilled (CPU)")
st.caption("Run with:  `streamlit run app.py`")

# ----- Models -----
class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x): return self.net(x)

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

def count_params(model: nn.Module):
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return None

def file_size_mb(path: Path):
    return (path.stat().st_size / 1e6) if path.exists() else None

@torch.no_grad()
def evaluate(model, loader, cap_batches=None):
    model.eval()
    correct, n = 0, 0
    for i, (x, y) in enumerate(loader):
        if cap_batches is not None and i == cap_batches: break
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        n += y.size(0)
    return correct / max(n, 1)

@torch.no_grad()
def latency_seconds(model, loader, repeats=1):
    model.eval()
    start = time.time()
    for _ in range(repeats):
        for x, _ in loader:
            _ = model(x.to(device))
    return (time.time() - start) / repeats

def get_test_loader(batch=512):
    tfm = transforms.Compose([transforms.ToTensor()])
    test = datasets.MNIST(root="./data", train=False, transform=tfm, download=True)
    return torch.utils.data.DataLoader(test, batch_size=batch, shuffle=False, num_workers=2)

def preprocess_uploaded(img: Image.Image):
    img = img.convert("L")
    img = ImageOps.invert(img)          # MNIST is white-on-black
    img = img.resize((28, 28))
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,28,28]

def predict_top3(model, x):
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()[0]
    top3 = probs.argsort()[-3:][::-1]
    return [(int(k), float(probs[k])) for k in top3]

# ----- Load artifacts -----
tpath = ART / "teacher_baseline.pth"
spath = ART / "student_distilled.pth"
qscript_path = ART / "teacher_quantized_scripted.pt"  # for real size comparison

teacher = TeacherNet().to(device)
student = StudentNet().to(device)

if not tpath.exists() or not spath.exists():
    st.error("Artifacts not found. Run `python prepare_artifacts.py` first.")
    st.stop()

teacher.load_state_dict(torch.load(tpath, map_location=device))
student.load_state_dict(torch.load(spath, map_location=device))

# Build quantized teacher (for accuracy/latency)
qteacher = torch.quantization.quantize_dynamic(
    copy.deepcopy(teacher).cpu(), {nn.Linear}, dtype=torch.qint8
)

# Persist a scripted quantized model (for *file size*), create once if missing
if not qscript_path.exists():
    scripted_q = torch.jit.script(qteacher)
    scripted_q.save(qscript_path)
q_size_mb = file_size_mb(qscript_path)

# ----- Sidebar -----
st.sidebar.title("Controls")
batch = st.sidebar.selectbox("Batch size (latency)", [1, 8, 64, 256], index=2)
cap = st.sidebar.selectbox("Cap test batches (faster)", [None, 20, 50, 100], index=2)

# ----- Metrics -----
st.subheader("1) Quick Evaluation & Latency")
test_loader = get_test_loader(batch=batch)

if st.button("Run metrics"):
    results = []

    acc_t = evaluate(teacher, test_loader, cap_batches=cap if cap is not None else None)
    lat_t = latency_seconds(teacher, test_loader, repeats=1)
    results.append(dict(Model="Baseline (Teacher)",
                        Params=count_params(teacher),
                        Size_MB=file_size_mb(tpath),
                        Test_Acc=round(acc_t, 4),
                        Latency_s=round(lat_t, 2)))

    acc_q = evaluate(qteacher, test_loader, cap_batches=cap if cap is not None else None)
    lat_q = latency_seconds(qteacher, test_loader, repeats=1)
    results.append(dict(Model="Quantized (Teacher int8)",
                        Params=count_params(teacher),  # same param count; stored smaller
                        Size_MB=q_size_mb,            # real size of scripted quantized file
                        Test_Acc=round(acc_q, 4),
                        Latency_s=round(lat_q, 2)))

    acc_s = evaluate(student, test_loader, cap_batches=cap if cap is not None else None)
    lat_s = latency_seconds(student, test_loader, repeats=1)
    results.append(dict(Model="Student (Distilled)",
                        Params=count_params(student),
                        Size_MB=file_size_mb(spath),
                        Test_Acc=round(acc_s, 4),
                        Latency_s=round(lat_s, 2)))

    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("Model")["Latency_s"], height=220)
    if df["Size_MB"].notna().any():
        st.bar_chart(df.set_index("Model")["Size_MB"], height=220)
    st.bar_chart(df.set_index("Model")["Test_Acc"], height=220)

# ----- Try an image -----
st.subheader("2) Try a Sample / Upload")
c1, c2 = st.columns(2)
with c1:
    test_loader_demo = get_test_loader(batch=1)
    sample_btn = st.button("Pick random test image")
with c2:
    uploaded = st.file_uploader("Upload digit (PNG/JPG)", type=["png", "jpg", "jpeg"])

img_tensor, img_show, label = None, None, None
if sample_btn:
    x, y = next(iter(test_loader_demo))
    img_tensor = x
    img_show = (x[0, 0].numpy() * 255).astype(np.uint8)
    label = int(y.item())
    st.image(img_show, caption=f"Ground truth: {label}", width=128)

if uploaded is not None:
    img = Image.open(uploaded)
    img_tensor = preprocess_uploaded(img)
    img_show = np.array(ImageOps.invert(img.convert("L")).resize((28, 28)))
    st.image(img_show, caption="Uploaded (normalized to MNIST style)", width=128)

if img_tensor is not None:
    st.markdown("**Predictions (Top-3):**")
    cols = st.columns(3)

    def show_preds(col, title, model):
        preds = predict_top3(model, img_tensor)
        dfp = pd.DataFrame(preds, columns=["Digit", "Probability"])
        dfp["Rank"] = [1, 2, 3]
        dfp["Probability"] = (dfp["Probability"] * 100).round(2).astype(str) + "%"
        dfp = dfp[["Rank", "Digit", "Probability"]]
        with col:
            st.caption(title); st.table(dfp)

    show_preds(cols[0], "Baseline", teacher)
    show_preds(cols[1], "Quantized", qteacher)
    show_preds(cols[2], "Student", student)

st.divider()
st.markdown("**Artifacts present:**")
st.code(
    f"teacher_baseline.pth: {'✅' if tpath.exists() else '❌'}\n"
    f"teacher_quantized_scripted.pt: {'✅' if qscript_path.exists() else '❌'}\n"
    f"student_distilled.pth: {'✅' if spath.exists() else '❌'}"
)
