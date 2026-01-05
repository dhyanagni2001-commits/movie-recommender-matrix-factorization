import numpy as np


def precision_recall_at_k(recommended, relevant, k):
    """
    recommended : list or array of recommended item indices in ranked order
    relevant    : set of ground-truth relevant item indices
    k           : cutoff rank
    """
    if k == 0:
        return 0.0, 0.0

    recommended_k = recommended[:k]

    hits = sum(1 for item in recommended_k if item in relevant)

    precision = hits / k
    recall = hits / max(1, len(relevant))

    return precision, recall


def ndcg_at_k(recommended, relevant, k):
    """
    Normalized Discounted Cumulative Gain at K.

    Binary relevance: 1 if item in relevant set, else 0.
    """
    if k == 0:
        return 0.0

    recommended_k = recommended[:k]

    gains = [1.0 if item in relevant else 0.0 for item in recommended_k]

    # DCG
    dcg = 0.0
    for idx, g in enumerate(gains):
        dcg += g / np.log2(idx + 2)  # rank index 0 -> log2(2)

    # IDCG (ideal DCG)
    ideal_gains = [1.0] * min(len(relevant), k)
    if not ideal_gains:
        return 0.0

    idcg = 0.0
    for idx, g in enumerate(ideal_gains):
        idcg += g / np.log2(idx + 2)

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)
