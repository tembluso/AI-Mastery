import sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# paths to your saved model
MODEL_DIR = "./finetuned-distilbert-imdb"

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tok, model

def predict(text):
    tok, model = load_model()
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        out = model(**enc)
        probs = F.softmax(out.logits, dim=-1).numpy()[0]
    return {"label": int(probs.argmax()), "prob_negative": float(probs[0]), "prob_positive": float(probs[1])}

if __name__ == "__main__":
    text = " ".join(sys.argv[1:])  # take CLI input
    if not text:
        print("Usage: python predict.py \"Your review here\"")
        sys.exit(1)
    print(predict(text))
