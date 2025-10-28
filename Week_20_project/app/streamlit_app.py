import os
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.config import INDEX_DIR, PROMPTS_DIR, CHAT_MODEL
from rag.ingest import ingest_paths
from rag.retriever import retrieve

load_dotenv()
client = OpenAI()

st.set_page_config(page_title="RAG Tweet Assistant (FAISS)", page_icon="🧠", layout="centered")
st.title("🧠 RAG Tweet Assistant — FAISS Edition")
st.caption("Upload notes → build FAISS index → generate grounded tweet/post drafts")

# --- Sidebar: settings
with st.sidebar:
    st.header("Settings")
    platform = st.selectbox("Platform", ["X/Twitter", "LinkedIn", "Threads", "Instagram"], index=0)
    variants = st.slider("# Variants", 1, 5, 3)
    max_chars = st.slider("Character budget", 120, 1000, 260, step=20)
    add_hashtags = st.checkbox("Suggest hashtags", value=False)
    add_emojis = st.checkbox("Suggest emojis", value=False)
    top_k = st.slider("Top-k chunks", 2, 12, 6)

# --- Uploader (optional ingestion)
uploaded_files = st.file_uploader(
    "Upload PDFs / TXT / MD (optional — or pre-index via CLI)",
    type=["pdf", "txt", "md", "markdown"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Index uploaded files"):
    tmp_dir = Path(".tmp_uploads")
    tmp_dir.mkdir(exist_ok=True)
    paths: List[Path] = []
    for uf in uploaded_files:
        p = tmp_dir / uf.name
        p.write_bytes(uf.getvalue())
        paths.append(p)
    n = ingest_paths(paths)
    st.success(f"Indexed {n} chunks. FAISS store at {INDEX_DIR}")

# --- Prompting UI
st.subheader("Generate a post")
query = st.text_area("What should the post be about? (keywords, topic, vibe)", height=80)
voice = st.text_input("Optional style/voice hints (e.g., 'curious, first-person, contrarian')")
call_to_action = st.text_input("Optional CTA (e.g., 'ask a question', 'share your take')")

if st.button("Generate drafts", type="primary"):
    if not query.strip():
        st.warning("Please enter a topic or query.")
        st.stop()

    # Retrieve
    try:
        hits = retrieve(query, top_k=top_k)
    except FileNotFoundError:
        st.info("No index found. Ingest notes first (sidebar upload button or CLI).")
        st.stop()

    if not hits:
        st.info("No context found. Try indexing notes first or broaden the query.")
        st.stop()

    # Build system prompt
    sys_path = PROMPTS_DIR / "tweet_system_prompt.txt"
    system_prompt = sys_path.read_text(encoding="utf-8")

    # Build user prompt with context
    context_blocks = []
    for h in hits:
        src = h["meta"].get("source_path", "unknown") if "meta" in h else "unknown"
        context_blocks.append(f"[Source: {src}]\n{h['text']}")
    joined_context = "\n\n---\n\n".join(context_blocks)

    extras = []
    if voice: extras.append(f"Voice: {voice}")
    if call_to_action: extras.append(f"CTA: {call_to_action}")
    if add_hashtags: extras.append("Include relevant, minimal hashtags.")
    if add_emojis: extras.append("Tasteful emojis allowed.")

    extras_text = "\n".join(extras) if extras else ""
    user_msg = f"""
    Write {variants} {platform} post variants, each ≤ {max_chars} characters.
    Ground EVERY line in the CONTEXT below. If context is weak/irrelevant, say so and stop.
    {extras_text}

    CONTEXT START
    {joined_context}
    CONTEXT END
    """

    with st.spinner("Calling the model…"):
        resp = client.chat.completions.create(
            model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
        )
        text = resp.choices[0].message.content

    st.divider()
    st.subheader("Drafts")
    st.write(text)

    # Save last result
    out_dir = Path("generated")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "drafts.txt"
    out_path.write_text(text, encoding="utf-8")
    st.download_button("Download drafts.txt", data=text, file_name="drafts.txt")
