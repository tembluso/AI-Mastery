
# Week 19 · Day 7 — Efficiency Showdown (CPU-only Demo)

### 🧠 Overview
This project compares three versions of the same neural network model trained on the MNIST dataset to demonstrate efficiency techniques for AI model deployment — all running **entirely on CPU**.

You’ll learn how quantization and distillation can drastically reduce model size and inference time while maintaining similar accuracy.

### ⚙️ Models Compared
| Model | Description | Goal |
|--------|--------------|------|
| **Teacher (Baseline)** | Original fully-trained model | Reference accuracy |
| **Quantized Teacher (int8)** | Reduced-precision version using dynamic quantization | Smaller, faster model |
| **Student (Distilled)** | Compact model trained to imitate the teacher | Similar performance with fewer parameters |

### 📊 What the App Does
- Loads the three models from `artifacts/`.
- Benchmarks **accuracy**, **model size**, and **latency** (seconds per test epoch).
- Displays metrics and bar charts in a clean Streamlit interface.
- Lets you upload your own digit image or pick a random test sample to see top‑3 predictions from each model.

### 🧩 Technologies Used
- **PyTorch** — for training, quantization, and distillation.  
- **Streamlit** — for building the interactive dashboard.  
- **TorchScript** — for exporting the quantized model as a deployable `.pt` file.

### 🚀 How to Run
```bash
# 1. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate model artifacts (fast CPU run)
python prepare_artifacts.py

# 4. Launch the app
streamlit run app.py
```

### 🧮 Expected Results
| Model | Size (MB) | Latency (s/test) | Accuracy |
|--------|------------|------------------|-----------|
| Baseline (Teacher) | ~2.5 MB | 6–8 s | ~0.96 |
| Quantized (int8) | ~0.8 MB | 3–5 s | ~0.95 |
| Student (Distilled) | ~1.2 MB | 2–4 s | ~0.93 |

*(Numbers will vary depending on CPU speed and subset size.)*

### 📁 Folder Structure
```
efficiency_showdown_app/
│
├── app.py                  # Streamlit demo
├── prepare_artifacts.py    # Fast CPU training + artifact generation
├── requirements.txt
└── artifacts/
    ├── teacher_baseline.pth
    ├── teacher_quantized_scripted.pt
    └── student_distilled.pth
```

### 🧾 Notes
- All models run **without GPU** — perfect for demonstrating deployment techniques on local machines.
- The quantized version reduces precision to int8 for faster CPU inference.
- The distilled student is smaller but still learns from the teacher’s soft outputs.

### 🧠 What I Learned
- How to benchmark speed/size trade‑offs in model deployment.  
- How quantization and distillation complement each other.  
- How to wrap experiments into a professional Streamlit dashboard.

---

**Author:** Federico Sánchez  
**Project:** AI Mastery · Week 19 · Day 7 — Scaling & Efficiency in AI
