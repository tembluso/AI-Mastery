import argparse, time, numpy as np, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

def build_tensors(dataset, tokenizer, max_len=128):
    enc = tokenizer(list(dataset['text']), padding='max_length', truncation=True, max_length=max_len)
    input_ids = torch.tensor(enc['input_ids'], dtype=torch.long)
    attention = torch.tensor(enc['attention_mask'], dtype=torch.long)
    labels = torch.tensor(dataset['label'], dtype=torch.long)
    return TensorDataset(input_ids, attention, labels)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention, labels in dataloader:
            input_ids = input_ids.to(device); attention = attention.to(device); labels = labels.to(device)
            out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
            logits = out.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return {'accuracy':acc, 'precision':prec, 'recall':rec, 'f1':f1}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='distilbert-base-uncased')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--train_samples', type=int, default=1000)
    ap.add_argument('--eval_samples', type=int, default=400)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--max_len', type=int, default=128)
    ap.add_argument('--outdir', default='./finetuned-distilbert-imdb')
    args = ap.parse_args()

    device = torch.device('cpu')
    torch.set_num_threads(max(1, torch.get_num_threads()))

    print('[CLS] Loading IMDB...')
    ds = load_dataset('imdb')
    tok = AutoTokenizer.from_pretrained(args.model)

    train_raw = ds['train'].shuffle(seed=42).select(range(min(args.train_samples, len(ds['train']))))
    eval_raw  = ds['test'].shuffle(seed=42).select(range(min(args.eval_samples, len(ds['test']))))

    train_ds = build_tensors(train_raw, tok, max_len=args.max_len)
    eval_ds  = build_tensors(eval_raw, tok, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader  = DataLoader(eval_ds, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optim_grouped = [
        {'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
        {'params':[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay':0.0},
    ]
    optimizer = AdamW(optim_grouped, lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps//10), num_training_steps=total_steps)

    print('[CLS] Training...')
    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        for step, (input_ids, attention, labels) in enumerate(train_loader, 1):
            input_ids = input_ids.to(device); attention = attention.to(device); labels = labels.to(device)
            out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            if step % 50 == 0:
                print(f'Epoch {epoch+1} Step {step}/{len(train_loader)} - loss {loss.item():.4f}')
        print(f'Epoch {epoch+1} time: {time.time()-t0:.1f}s')
        metrics = evaluate(model, eval_loader, device)
        print('Eval:', metrics)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    print('[CLS] Saved to', outdir)
