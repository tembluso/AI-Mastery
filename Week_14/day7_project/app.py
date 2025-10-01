# Streamlit app — CPU-friendly, optional tiny bootstrap, no Trainer-specific args
import time, numpy as np, torch, streamlit as st
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW


CLASS_DIR = Path('./finetuned-distilbert-imdb')
GEN_DIR   = Path('./finetuned-gpt2-imdb')

st.set_page_config(page_title='Fine-Tuned LLM Demo (CPU)', page_icon='🤖', layout='centered')
st.title('Week 14 · Day 7 — Fine-Tuned LLM Demo (CPU)')
st.caption('Classification (DistilBERT) + optional Generation (GPT-2).')

@st.cache_resource
def load_classifier():
    tok = AutoTokenizer.from_pretrained(CLASS_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CLASS_DIR)
    model.eval(); return tok, model

@st.cache_resource
def load_generator():
    tok = AutoTokenizer.from_pretrained(GEN_DIR)
    model = AutoModelForCausalLM.from_pretrained(GEN_DIR)
    model.eval(); return tok, model

def build_cls_tensors(dataset, tokenizer, max_len=128):
    enc = tokenizer(list(dataset['text']), padding='max_length', truncation=True, max_length=max_len)
    input_ids = torch.tensor(enc['input_ids'], dtype=torch.long)
    attention = torch.tensor(enc['attention_mask'], dtype=torch.long)
    labels = torch.tensor(dataset['label'], dtype=torch.long)
    return TensorDataset(input_ids, attention, labels)

def eval_cls(model, dataloader, device):
    model.eval(); all_preds=[]; all_labels=[]
    with torch.no_grad():
        for input_ids, attention, labels in dataloader:
            input_ids=input_ids.to(device); attention=attention.to(device); labels=labels.to(device)
            out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
            preds = torch.argmax(out.logits, dim=-1)
            all_preds.append(preds.cpu()); all_labels.append(labels.cpu())
    y_pred = torch.cat(all_preds).numpy(); y_true=torch.cat(all_labels).numpy()
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return {'accuracy':acc, 'precision':prec, 'recall':rec, 'f1':f1}

def tiny_bootstrap_classifier(train_samples=600, eval_samples=200, epochs=1, model_name='distilbert-base-uncased'):
    st.info('Bootstrapping classifier (tiny, CPU)...')
    ds = load_dataset('imdb')
    tok = AutoTokenizer.from_pretrained(model_name)
    train_raw = ds['train'].shuffle(seed=42).select(range(min(train_samples, len(ds['train']))))
    eval_raw  = ds['test'].shuffle(seed=42).select(range(min(eval_samples, len(ds['test']))))
    train_ds = build_cls_tensors(train_raw, tok, max_len=128)
    eval_ds  = build_cls_tensors(eval_raw, tok, max_len=128)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    eval_loader  = DataLoader(eval_ds, batch_size=8)

    device = torch.device('cpu'); torch.set_num_threads(max(1, torch.get_num_threads()))
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optim_grouped = [
        {'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
        {'params':[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay':0.0},
    ]
    optimizer = AdamW(optim_grouped, lr=2e-5)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1,total_steps//10), total_steps)

    model.train()
    for epoch in range(epochs):
        t0=time.time()
        for step, (input_ids, attention, labels) in enumerate(train_loader,1):
            input_ids=input_ids.to(device); attention=attention.to(device); labels=labels.to(device)
            out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
            loss = out.loss; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            if step % 50 == 0: st.write(f'Epoch {epoch+1} Step {step}/{len(train_loader)} — loss {loss.item():.4f}')
        st.write(f'Epoch {epoch+1} time: {time.time()-t0:.1f}s')
        st.write('Eval:', eval_cls(model, eval_loader, device))

    CLASS_DIR.mkdir(parents=True, exist_ok=True); model.save_pretrained(CLASS_DIR); tok.save_pretrained(CLASS_DIR)
    st.success(f'Saved classifier to {CLASS_DIR.resolve()}')

def tiny_bootstrap_generator(train_samples=800, eval_samples=200, epochs=1, model_name='gpt2'):
    st.info('Bootstrapping GPT-2 (tiny, CPU)...')
    ds = load_dataset('imdb')
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    train_raw = ds['train'].shuffle(seed=42).select(range(min(train_samples, len(ds['train']))))
    eval_raw  = ds['test'].shuffle(seed=42).select(range(min(eval_samples, len(ds['test']))))

    def to_tensor(dataset, max_len=128):
        enc = tok(list(dataset['text']), padding='max_length', truncation=True, max_length=max_len)
        input_ids = torch.tensor(enc['input_ids'], dtype=torch.long)
        attention = torch.tensor(enc['attention_mask'], dtype=torch.long)
        labels = input_ids.clone()
        return TensorDataset(input_ids, attention, labels)

    train_ds = to_tensor(train_raw); eval_ds = to_tensor(eval_raw)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    eval_loader  = DataLoader(eval_ds, batch_size=2)

    device = torch.device('cpu'); model = AutoModelForCausalLM.from_pretrained(model_name); model.resize_token_embeddings(len(tok)); model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1,total_steps//10), total_steps)

    for epoch in range(epochs):
        t0=time.time(); model.train()
        for step, (input_ids, attention, labels) in enumerate(train_loader,1):
            input_ids=input_ids.to(device); attention=attention.to(device); labels=labels.to(device)
            out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
            loss=out.loss; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            if step % 50 == 0: st.write(f'[GEN] Epoch {epoch+1} Step {step}/{len(train_loader)} — loss {loss.item():.4f}')
        st.write(f'[GEN] Epoch {epoch+1} time: {time.time()-t0:.1f}s')
        # quick eval loss
        model.eval(); total, n = 0.0, 0
        with torch.no_grad():
            for input_ids, attention, labels in eval_loader:
                input_ids=input_ids.to(device); attention=attention.to(device); labels=labels.to(device)
                out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
                total += float(out.loss.item()) * input_ids.size(0); n += input_ids.size(0)
        st.write('[GEN] Eval loss:', total/max(1,n))

    GEN_DIR.mkdir(parents=True, exist_ok=True); model.save_pretrained(GEN_DIR); tok.save_pretrained(GEN_DIR)
    st.success(f'Saved generator to {GEN_DIR.resolve()}')

with st.sidebar:
    st.header('Bootstrap (tiny, CPU)')
    if st.button('Bootstrap Classifier'):
        tiny_bootstrap_classifier()
    if st.button('Bootstrap Generator'):
        tiny_bootstrap_generator()

tabs = st.tabs(['🔎 Sentiment Classification', '✍️ Review-Style Generation'])

with tabs[0]:
    st.subheader('DistilBERT Sentiment (IMDB)')
    if not CLASS_DIR.exists() or not any(CLASS_DIR.iterdir()):
        st.warning('Classifier not found at ./finetuned-distilbert-imdb. Use the sidebar to bootstrap or run train_cls.py.')
    else:
        tok, model = load_classifier()
        col1, col2 = st.columns(2)
        with col1:
            text = st.text_area('Enter text', 'This movie was surprisingly good!')
            if st.button('Predict', key='pred1'):
                t0=time.time()
                enc = tok([text], return_tensors='pt', padding=True, truncation=True, max_length=128)
                with torch.no_grad():
                    out = model(**enc)
                    probs = torch.softmax(out.logits, dim=-1).numpy()[0]
                pred = int(np.argmax(probs)); latency=(time.time()-t0)*1000
                st.success(f'Label: **{model.config.id2label.get(pred, pred)}** | neg={probs[0]:.3f}, pos={probs[1]:.3f} | {latency:.1f} ms')
        with col2:
            batch = st.text_area('Batch (one per line)', 'I hated the ending.\nAbsolutely wonderful!')
            if st.button('Batch Predict', key='pred2'):
                lines = [s for s in batch.splitlines() if s.strip()]
                t0=time.time()
                enc = tok(lines, return_tensors='pt', padding=True, truncation=True, max_length=128)
                with torch.no_grad():
                    out = model(**enc)
                    probs = torch.softmax(out.logits, dim=-1).numpy()
                latency=(time.time()-t0)*1000
                st.info(f'Batch size {len(lines)} | {latency:.1f} ms')
                for s,p in zip(lines, probs):
                    st.write(f'- **{model.config.id2label.get(int(p.argmax()), int(p.argmax()))}** | neg={p[0]:.3f}, pos={p[1]:.3f} — _{s[:80]}_')

with tabs[1]:
    st.subheader('GPT-2 Review-Style Generation (optional)')
    if not GEN_DIR.exists() or not any(GEN_DIR.iterdir()):
        st.warning('Generator not found at ./finetuned-gpt2-imdb. Use the sidebar to bootstrap or run train_gen.py.')
    else:
        tok, gen_model = load_generator()
        prompt = st.text_input('Prompt', 'The film was')
        c1, c2, c3 = st.columns(3)
        with c1: max_new = st.number_input('max_new_tokens', min_value=16, max_value=256, value=64, step=8)
        with c2: temperature = st.slider('temperature', 0.2, 2.0, 0.8, 0.05)
        with c3: top_p = st.slider('top_p', 0.1, 1.0, 0.95, 0.05)
        if st.button('Generate'):
            enc = tok(prompt, return_tensors='pt')
            t0=time.time()
            with torch.no_grad():
                out = gen_model.generate(**enc, max_new_tokens=int(max_new), do_sample=True, temperature=float(temperature), top_p=float(top_p), pad_token_id=tok.eos_token_id)
            txt = tok.decode(out[0], skip_special_tokens=True)
            st.code(txt); st.caption(f'Latency: {(time.time()-t0)*1000:.1f} ms')
