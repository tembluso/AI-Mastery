import csv, argparse

def compute_metrics(path: str, k: int = 6):
    rows = []
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    by_q = {}
    for row in rows:
        by_q.setdefault(row["query"], []).append(int(row["relevant"]))
    for q, rels in by_q.items():
        rels = rels[:k]
        total_rel = max(1, sum(rels))
        p_at_k = sum(rels) / len(rels) if rels else 0.0
        r_at_k = sum(rels) / total_rel if total_rel else 0.0
        print(f"{q}: P@{k}={p_at_k:.3f}  R@{k}={r_at_k:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    compute_metrics(args.labels_csv, args.k)
