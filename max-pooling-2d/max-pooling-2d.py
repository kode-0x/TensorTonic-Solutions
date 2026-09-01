def max_pooling_2d(X: list, pool_size: int) -> list:
    """
    Returns non-overlapping maximum-pooled windows.
    """
    # Write code here
    h = len(X)
    w = len(X[0])

    output = []

    for i in range(0, h - pool_size + 1, pool_size):
        row = []
        for j in range(0, w - pool_size + 1, pool_size):
            maximum = X[i][j]

            for a in range(pool_size):
                for b in range(pool_size):
                    maximum = max(maximum, X[i + a][j + b])

            row.append(maximum)

        output.append(row)

    return output