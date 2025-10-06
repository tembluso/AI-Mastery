import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")

def preprocess_canvas(rgba_array, add_dilate=False):
    # 1) to grayscale, invert to MNIST style (white digit on black)
    img = Image.fromarray((rgba_array[:, :, :3]).astype("uint8"))
    img = ImageOps.grayscale(img)
    img = ImageOps.invert(img)

    # 2) hard threshold to kill background noise
    a = np.array(img).astype(np.float32)
    a[a < 30] = 0.0  # keep strokes only

    # 3) if empty canvas, return zeros
    ys, xs = np.where(a > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((28, 28), dtype=np.float32)

    # 4) crop to bounding box
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cropped = a[y0:y1+1, x0:x1+1]

    # 5) (optional) slightly dilate to help thin strokes
    if add_dilate:
        pil_c = Image.fromarray(cropped)
        pil_c = pil_c.filter(ImageFilter.MaxFilter(size=3))
        cropped = np.array(pil_c).astype(np.float32)

    # 6) pad to square with margin
    h, w = cropped.shape
    size = max(h, w) + 8  # margin like MNIST cropping
    square = np.zeros((size, size), dtype=np.float32)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = cropped

    # 7) center by center-of-mass shift (like classic MNIST preprocess)
    ys, xs = np.nonzero(square)
    cy, cx = ys.mean(), xs.mean()
    sy, sx = np.array(square.shape) / 2.0
    dy, dx = int(round(sy - cy)), int(round(sx - cx))
    square = np.roll(square, shift=(dy, dx), axis=(0, 1))

    # 8) resize to 28x28 with slight blur to anti-alias thick lines
    pil_sq = Image.fromarray(square.astype(np.uint8)).convert("L").filter(ImageFilter.GaussianBlur(radius=0.5))
    img28 = pil_sq.resize((28, 28), Image.BILINEAR)

    # 9) normalize to [0,1] and light re-threshold
    arr = np.array(img28).astype(np.float32) / 255.0
    arr = (arr > 0.02) * arr
    return arr


# -------------------- Model --------------------
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

@st.cache_resource(show_spinner=True)
def get_model(device="cpu"):
    # Keep for inference use if you like, but DO NOT train this instance
    return DigitNet().to(device)

def load_or_train(model_path="mnist_mlp.pt", device="cpu", epochs=3):
    # Always use a fresh model for load/train to avoid in-place/version issues
    model = DigitNet().to(device)

    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model, False
    except Exception:
        pass

    # Train quick baseline
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        st.write(f"Epoch {epoch+1}/{epochs} — loss: {running/len(train_loader):.4f}")

    torch.save(model.state_dict(), model_path)
    model.eval()
    return model, True


# -------------------- UI --------------------
st.title("🖊️ MNIST Digit Recognizer — Week 5 Mini-Project")
st.caption("Draw a digit (0–9) and the model will predict it. If no weights are found, it will train (~1–2 min).")

col1, col2 = st.columns([3,2])
with col1:
    st.subheader("Draw here")
    bg_color = "#FFFFFF"
    stroke_color = st.color_picker("Stroke color", "#000000")
    stroke_width = st.slider("Stroke width", 10, 40, 20)
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=280, width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Controls")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Device: **{device}**")
    model, trained_now = load_or_train(device=device)
    if trained_now:
        st.success("Trained a fresh model and cached weights.")

    if st.button("Predict", use_container_width=True):
        if canvas_result.image_data is None:
            st.warning("Please draw a digit first.")
        else:
            # Preprocess the canvas into a MNIST-like 28x28 array
            arr = preprocess_canvas(canvas_result.image_data, add_dilate=False)  # try True if strokes are too thin
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))

            st.subheader(f"Prediction: **{pred}**")
            top3_idx = probs.argsort()[-3:][::-1]
            st.write("Top-3 probabilities:")
            for i in top3_idx:
                st.write(f"{i}: {probs[i]:.3f}")

            st.image(Image.fromarray((arr*255).astype(np.uint8)),
                     caption="Model input (28×28)", width=140)

st.caption("Tips: draw large, centered strokes. This is a simple MLP; a CNN will improve accuracy (Week 6).")
