def rating_normalization(matrix: list) -> list:
    """
    Returns the mean-centered user-item matrix.
    """
    normalized = []

    for row in matrix:
        ratings = [x for x in row if x != 0]

        if not ratings:
            normalized.append([0.0] * len(row))
            continue

        mean = sum(ratings) / len(ratings)

        normalized.append([
            float(x - mean) if x != 0 else 0.0
            for x in row
        ])

    return normalized