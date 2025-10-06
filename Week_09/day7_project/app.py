# app.py
import streamlit as st
import pandas as pd
from predict import load_bundle, predict_text

st.set_page_config(page_title="IMDB Sentiment — Week 9", page_icon="🎬")
st.title("🎬 IMDB Sentiment — Baseline LSTM (Fresh Train)")
st.caption("Trains via train.py → produces bundle.pt → this app loads that bundle for inference.")

@st.cache_resource
def _load_bundle():
    # Ensure bundle.pt is in the same folder (run train.py first)
    return load_bundle("bundle.pt")

model, tok, vocab, device, pad_idx, unk_idx, max_len = _load_bundle()

tab1, tab2 = st.tabs(["Single Review", "Batch"])

with tab1:
    text = st.text_area("Paste a movie review:", height=180, placeholder="I loved this movie!")
    thr = st.slider("Uncertainty threshold", 0.50, 0.95, 0.60, 0.01)
    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            out = predict_text(text, model, tok, vocab, device, pad_idx, unk_idx, max_len, return_probs=True)
            label = out["label"] if out["prob"] >= thr else "Uncertain"
            st.metric("Prediction", label, delta=f"{out['prob']:.2%}")
            st.write("Class probabilities:", out["probs"])

with tab2:
    st.write("One review per line or upload a .txt file.")
    uploaded = st.file_uploader("TXT file", type=["txt"])
    pasted = st.text_area("Or paste lines here:", height=150)
    if st.button("Run Batch"):
        lines = []
        if uploaded is not None:
            content = uploaded.read().decode("utf-8", errors="ignore")
            lines += [ln.strip() for ln in content.splitlines() if ln.strip()]
        lines += [ln.strip() for ln in pasted.splitlines() if ln.strip()]

        if not lines:
            st.info("No text provided.")
        else:
            rows = []
            for i, ln in enumerate(lines, 1):
                out = predict_text(ln, model, tok, vocab, device, pad_idx, unk_idx, max_len, return_probs=True)
                rows.append({
                    "#": i,
                    "Label": out["label"],
                    "Confidence": f"{out['prob']:.2%}",
                    "neg": f"{out['probs']['neg']:.2%}",
                    "pos": f"{out['probs']['pos']:.2%}",
                    "Text": ln[:120] + ("..." if len(ln) > 120 else "")
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name="predictions.csv", mime="text/csv")

st.divider()
with st.expander("About"):
    st.markdown("""
- **Single-file bundle** (`bundle.pt`) contains: state dict, vocab, `<PAD>/<UNK>`, dims, tokenizer, max_len.  
- Consistent preprocessing prevents label inversions and weird outputs.  
- Use the **threshold** slider to mark low-confidence as *Uncertain*.  
""")
