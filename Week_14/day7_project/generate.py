from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from pathlib import Path
import torch

MODEL_DIR = Path('./finetuned-gpt2-imdb')

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model.eval()

def generate(prompt: str, max_new_tokens=64, temperature=0.8, top_p=0.95):
    enc = tok(prompt, return_tensors='pt')
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=top_p,
            pad_token_id=tok.eos_token_id
        )
    return tok.decode(out[0], skip_special_tokens=True)

if __name__ == '__main__':
    prompt = " ".join(sys.argv[1:]) or "The film was"
    print(generate(prompt))
