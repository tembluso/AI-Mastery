#!/usr/bin/env python3
# CLI for Mini GPT vs BERT Playground

import argparse, json
from transformers import pipeline

def gpt_complete(prompt: str, max_new_tokens=40, temperature=1.0, top_k=50, top_p=0.95, n=1):
    gen = pipeline("text-generation", model="gpt2")
    outs = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
        top_p=top_p,
        num_return_sequences=n,
        do_sample=True,
        pad_token_id=gen.tokenizer.eos_token_id,
    )
    return [o["generated_text"] for o in outs]

def bert_fill(masked: str, top_k=5):
    fill = pipeline("fill-mask", model="bert-base-uncased")
    res = fill(masked, top_k=top_k)
    if isinstance(res, dict):
        res = [res]
    return res

def main():
    p = argparse.ArgumentParser(description="Mini GPT vs BERT Playground (CLI)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gpt", help="GPT-2 continuation")
    g.add_argument("prompt", type=str)
    g.add_argument("--max_new_tokens", type=int, default=40)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top_k", type=int, default=50)
    g.add_argument("--top_p", type=float, default=0.95)
    g.add_argument("-n", "--num_return_sequences", type=int, default=2)

    b = sub.add_parser("bert", help="BERT fill-mask")
    b.add_argument("masked", type=str, help="Sentence containing [MASK]")
    b.add_argument("--top_k", type=int, default=5)

    args = p.parse_args()

    if args.cmd == "gpt":
        outs = gpt_complete(
            args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.num_return_sequences
        )
        print(json.dumps({"prompt": args.prompt, "completions": outs}, ensure_ascii=False, indent=2))
    else:
        outs = bert_fill(args.masked, args.top_k)
        print(json.dumps({"masked": args.masked, "predictions": outs}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
