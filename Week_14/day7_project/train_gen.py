import argparse, time, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW


def build_tensors_text(dataset, tok, max_len=128):
    enc = tok(list(dataset['text']), padding='max_length', truncation=True, max_length=max_len)
    input_ids = torch.tensor(enc['input_ids'], dtype=torch.long)
    attention = torch.tensor(enc['attention_mask'], dtype=torch.long)
    # For causal LM, labels are input_ids shifted internally by model; many models accept labels=input_ids directly
    labels = input_ids.clone()
    return TensorDataset(input_ids, attention, labels)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt2')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--train_samples', type=int, default=1000)
    ap.add_argument('--eval_samples', type=int, default=400)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--max_len', type=int, default=128)
    ap.add_argument('--outdir', default='./finetuned-gpt2-imdb')
    args = ap.parse_args()

    device = torch.device('cpu')
    torch.set_num_threads(max(1, torch.get_num_threads()))

    print('[GEN] Loading IMDB...')
    ds = load_dataset('imdb')
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_raw = ds['train'].shuffle(seed=42).select(range(min(args.train_samples, len(ds['train']))))
    eval_raw  = ds['test'].shuffle(seed=42).select(range(min(args.eval_samples, len(ds['test']))))

    train_ds = build_tensors_text(train_raw, tok, max_len=args.max_len)
    eval_ds  = build_tensors_text(eval_raw, tok, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader  = DataLoader(eval_ds, batch_size=args.batch_size)

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tok))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1,total_steps//10), num_training_steps=total_steps)

    print('[GEN] Training... (tiny, CPU)')
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
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
        # quick eval loss
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for input_ids, attention, labels in eval_loader:
                input_ids = input_ids.to(device); attention = attention.to(device); labels = labels.to(device)
                out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
                tot += float(out.loss.item()) * input_ids.size(0); n += input_ids.size(0)
        print('Eval loss:', tot/max(1,n))

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    print('[GEN] Saved to', outdir)
