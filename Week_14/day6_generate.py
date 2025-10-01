import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "./finetuned-gpt2-imdb"

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    model.eval()
    return tok, model

def generate(prompt, max_new_tokens=40, temperature=0.8, top_p=0.95):
    tok, model = load_model()
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=temperature,
                             top_p=top_p, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:])
    if not prompt:
        print("Usage: python generate.py \"The film reminded me of...\"")
        sys.exit(1)
    print(generate(prompt))
