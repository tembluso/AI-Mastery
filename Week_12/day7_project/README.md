# Mini-Transformer Demo — Week 12 · Day 7

A tiny encoder–decoder Transformer trained on a **reverse-numbers** toy task.  
The Streamlit app lets you train, evaluate, predict, and visualize **decoder cross-attention**.

## Project Structure
- `app.py` – Streamlit app (train, evaluate, predict, visualize attention)
- `train.py` – optional standalone trainer that saves `checkpoint.pt`
- `requirements.txt` – dependencies for the app
- `README.md` – this file

## Setup
```bash
python -m venv venv
# Activate the environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Run the app
```bash
streamlit run app.py
```

## Usage
- Train or load a checkpoint inside the app.
- Evaluate test set accuracy + BLEU-1.
- Input your own sequence (10 integers between 1–19) and see predicted vs reference output.
- Visualize cross-attention heatmaps (expect **anti-diagonal** for reverse task).

## Typical Results
- Test Accuracy: ≥ 90% after ~10–15 epochs
- BLEU-1: ≥ 90%

## Notes
- Cross-attention visualization uses trained decoder weights (`need_weights=True`).
- Default hyperparameters: `d_model=64`, `num_layers=2`, `num_heads=4`.
