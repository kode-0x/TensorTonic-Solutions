def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    sums = {}
    counts = {}

    for category, target in zip(categories, targets):
        sums[category] = sums.get(category, 0) + target
        counts[category] = counts.get(category, 0) + 1

    return [float(sums[c] / counts[c]) for c in categories]
