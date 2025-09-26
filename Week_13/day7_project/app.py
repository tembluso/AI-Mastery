#!/usr/bin/env python3
# Week 13 · Day 7 — Mini GPT vs BERT Playground
# Streamlit app: GPT continuation vs BERT fill-mask

import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Mini GPT vs BERT Playground", layout="centered")

st.title("🧠 Mini GPT vs BERT Playground")
st.caption("Type a sentence → GPT continues it, or add [MASK] → BERT fills it.")

with st.sidebar:
    st.header("⚙️ Generation Settings (GPT)")
    max_new_tokens = st.slider("max_new_tokens", 1, 200, 40, 1)
    temperature = st.slider("temperature", 0.1, 1.5, 1.0, 0.1)
    top_k = st.slider("top_k", 0, 100, 50, 1)
    top_p = st.slider("top_p", 0.0, 1.0, 0.95, 0.05)
    num_return_sequences = st.slider("num_return_sequences", 1, 5, 2, 1)

@st.cache_resource(show_spinner=True)
def load_pipelines():
    gen = pipeline("text-generation", model="gpt2")
    fill = pipeline("fill-mask", model="bert-base-uncased")
    return gen, fill

gen, fill = load_pipelines()

example_prompts = [
    "The future of AI is",
    "In 10 years, students will",
    "Once upon a time,"
]
example_masks = [
    "The capital of France is [MASK].",
    "AI is changing the [MASK].",
    "The [MASK] barked loudly."
]

st.subheader("✍️ Input")
user_text = st.text_area(
    "Enter text. If it contains [MASK], BERT will try to fill it. Otherwise, GPT will continue.",
    value=example_prompts[0],
    height=120,
)

def run_gpt(prompt: str):
    outputs = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        pad_token_id=gen.tokenizer.eos_token_id,
    )
    return [o["generated_text"] for o in outputs]

def run_bert(masked: str, topk: int = 5):
    results = fill(masked, top_k=topk)
    # HF returns a list of dicts; normalize shape (list) if single prediction
    if isinstance(results, dict):
        results = [results]
    return results

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Quick Examples (GPT)**")
    if st.button("Use: “The future of AI is”"):
        user_text = "The future of AI is"
    if st.button("Use: “In 10 years, students will”"):
        user_text = "In 10 years, students will"
    if st.button("Use: “Once upon a time,”"):
        user_text = "Once upon a time,"
with col2:
    st.markdown("**Quick Examples (BERT)**")
    if st.button("Use: “The capital of France is [MASK].”"):
        user_text = "The capital of France is [MASK]."
    if st.button("Use: “AI is changing the [MASK].”"):
        user_text = "AI is changing the [MASK]."
    if st.button("Use: “The [MASK] barked loudly.”"):
        user_text = "The [MASK] barked loudly."

st.write("---")

if st.button("🚀 Run"):
    if "[MASK]" in user_text:
        st.markdown("### 🤖 BERT Fill‑Mask Results")
        try:
            preds = run_bert(user_text, topk=5)
            for i, p in enumerate(preds, 1):
                seq = p.get("sequence", "")
                token_str = p.get("token_str", "")
                score = p.get("score", 0.0)
                st.write(f"**{i}.** `{token_str}` — score: `{score:.4f}`")
                st.write(seq)
                st.write("")
        except Exception as e:
            st.error(f"Fill-mask failed: {e}")
    else:
        st.markdown("### ✨ GPT Continuations")
        try:
            outs = run_gpt(user_text)
            for i, txt in enumerate(outs, 1):
                st.write(f"**Completion {i}:**")
                st.write(txt)
                st.write("")
        except Exception as e:
            st.error(f"Generation failed: {e}")

st.write("---")
with st.expander("ℹ️ What’s happening under the hood?"):
    st.markdown(
        """
**GPT-2** (decoder-only) is trained with **causal language modeling**: predict the *next* token using only left context.  
**BERT** (encoder-only) is trained with **masked language modeling**: recover hidden tokens like `[MASK]` using both left and right context.

Use GPT when you want open-ended **generation** (stories, continuations).  
Use BERT when you need **understanding** or missing-word inference (classification, embeddings, fill‑mask).
"""
    )
