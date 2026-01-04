import numpy as np


def precision_recall_at_k(recommended, relevant, k):
    """
    recommended : list or array of recommended item indices in ranked order
    relevant    : set of ground-truth relevant item indices
    k           : cutoff rank
    """

    # take only top-k recommended
    recommended_k = recommended[:k]

    # count hits (items that are actually relevant)
    hits = sum(1 for item in recommended_k if item in relevant)

    # precision = hits among recommended
    precision = hits / k

    # recall = hits among relevant
    recall = hits / max(1, len(relevant))

    return precision, recall
