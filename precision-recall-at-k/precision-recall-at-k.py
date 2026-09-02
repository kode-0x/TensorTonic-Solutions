def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    relevant_set = set(relevant)
    hits = sum(item in relevant_set for item in recommended[:k])

    precision = hits / k
    recall = hits / len(relevant)

    return [precision, recall]