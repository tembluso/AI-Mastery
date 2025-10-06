# app/frontend.py
import streamlit as st
import subprocess, sys, os
import pandas as pd

from backend import answer_query  # script-style import (same folder)

st.set_page_config(page_title="Q&A Bot", page_icon="💬", layout="centered")
st.title("💬 Q&A Bot with Citations")
st.caption("Answers are grounded in your local .txt/.md documents.")

# --- Sidebar controls ---
with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider("Top-k documents", 1, 10, 3)

    if st.button("⚙️ Rebuild index"):
        cmd = [sys.executable, "app/ingestion.py",
               "--data_dir", "data/sample_docs",
               "--out_dir", "vector_store"]
        with st.spinner("Building index..."):
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                st.success("Index rebuilt.")
                st.code(out)
            except subprocess.CalledProcessError as e:
                st.error("Failed to build index:")
                st.code(e.output)

# --- Chat history state (single source of truth) ---
# Each item: {"role": "user"|"assistant", "content": str, "citations": list|None}
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 1) Render existing history (ONLY once) ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("📚 Sources"):
                for c in msg["citations"]:
                    st.markdown(f"**{c['ref']}** — *{c['doc_id']}*, chunk {c['chunk_id']}")
                    st.markdown(f"> {c['excerpt']}...")

# --- 2) New user input ---
prompt = st.chat_input("Ask a question...")
if prompt:
    # append + show user message
    st.session_state.messages.append({"role": "user", "content": prompt, "citations": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # get answer
    with st.spinner("Thinking..."):
        resp = answer_query(prompt, k=top_k)

    # build assistant content with inline markers
    markers = " ".join([f"[{i+1}]" for i in range(len(resp["citations"]))])
    answer_text = resp["answer"] + (f"\n\n{markers}" if markers else "")

    # append + show assistant message (store citations so re-render keeps them)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "citations": resp["citations"]}
    )
    with st.chat_message("assistant"):
        st.markdown(answer_text)
        if resp["citations"]:
            with st.expander("📚 Sources"):
                for c in resp["citations"]:
                    st.markdown(f"**{c['ref']}** — *{c['doc_id']}*, chunk {c['chunk_id']}")
                    st.markdown(f"> {c['excerpt']}...")



