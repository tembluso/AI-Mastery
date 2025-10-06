# app/evaluate.py
import numpy as np
from backend import answer_query, load_vector_store, retrieve

def eval_retrieval(qa_rows, k=3):
    vectorizer, matrix, nn, chunks = load_vector_store()
    hits = []
    for r in qa_rows:
        res = retrieve(r["question"], k, vectorizer, nn, chunks)
        blob = " ".join(res["text"].tolist()).lower()
        found = int(any(w in blob for w in r["answer_ref"].lower().split()))
        hits.append(found)
    return float(np.mean(hits))

def eval_generation(qa_rows):
    errs = 0
    for r in qa_rows:
        ans = answer_query(r["question"])["answer"].lower()
        if r["answer_ref"].lower() not in ans:
            errs += 1
    return errs / len(qa_rows)

def main():
    qa_rows = [
        {"question": "What is chunk overlap?", "answer_ref": "overlap"},
        {"question": "Why use citations?", "answer_ref": "citations"},
        {"question": "What does RAG combine?", "answer_ref": "retrieval"},
    ]
    p3 = eval_retrieval(qa_rows, k=3)
    hall = eval_generation(qa_rows)
    print(f"Precision@3 = {p3:.2f} | Hallucination Rate = {hall:.2f}")

if __name__ == "__main__":
    main()
