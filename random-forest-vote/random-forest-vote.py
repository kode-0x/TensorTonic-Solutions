def random_forest_vote(predictions: list) -> list:
    """
    Returns the majority-vote label for every sample.
    """
    result = []

    for votes in zip(*predictions):
        counts = {}
        for label in votes:
            counts[label] = counts.get(label, 0) + 1

        result.append(min(counts, key=lambda x: (-counts[x], x)))

    return result