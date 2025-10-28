from __future__ import annotations
from typing import Sequence
import numpy as np

def precision_at_k(pred_relevances: Sequence[int], k: int) -> float:
    k = min(k, len(pred_relevances))
    if k == 0: 
        return 0.0
    return float(np.sum(pred_relevances[:k])) / k

def recall_at_k(pred_relevances: Sequence[int], total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    k = min(k, len(pred_relevances))
    return float(np.sum(pred_relevances[:k])) / total_relevant

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 2: a = a[0]
    if b.ndim == 2: b = b[0]
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))
