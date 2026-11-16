def mean_rating_imputation(ratings_matrix: list, mode: str) -> list:
    """
    Returns a copy with missing ratings replaced by user or item means.
    """
    rows = len(ratings_matrix)
    cols = len(ratings_matrix[0])

    result = [row[:] for row in ratings_matrix]

    if mode == "user":
        for i in range(rows):
            ratings = [x for x in ratings_matrix[i] if x != 0]
            mean = sum(ratings) / len(ratings) if ratings else 0.0

            for j in range(cols):
                if result[i][j] == 0:
                    result[i][j] = float(mean)

    elif mode == "item":
        for j in range(cols):
            ratings = [
                ratings_matrix[i][j]
                for i in range(rows)
                if ratings_matrix[i][j] != 0
            ]
            mean = sum(ratings) / len(ratings) if ratings else 0.0

            for i in range(rows):
                if result[i][j] == 0:
                    result[i][j] = float(mean)

    return result