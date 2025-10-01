from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, sys, json, numpy as np
from pathlib import Path

MODEL_DIR = Path('./finetuned-distilbert-imdb')

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def predict(text: str):
    enc = tok(text, return_tensors='pt', truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).numpy()[0]
    label_id = int(np.argmax(probs))
    label = model.config.id2label.get(label_id, label_id)
    return {'label': label, 'probability': float(probs[label_id])}

if __name__ == '__main__':
    text = " ".join(sys.argv[1:]) or "This movie was fantastic!"
    print(json.dumps(predict(text), indent=2))
