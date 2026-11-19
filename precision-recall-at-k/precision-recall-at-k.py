def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    top_k = recommended[:k]                       # first k, in the given order
    rel = set(relevant)
    hits = [x for x in top_k if x in rel]
    precision = len(hits) / k if k else 0.0
    recall = len(hits) / len(rel) if rel else 0.0
    return [precision, recall]