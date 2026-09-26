def maxpool_forward(X: list, pool_size: int, stride: int) -> list:
    """
    Returns the maximum value from every pooling window.
    """
    H = len(X)
    W = len(X[0])

    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    output = []

    for i in range(H_out):
        row = []

        for j in range(W_out):
            max_value = float('-inf')

            for a in range(pool_size):
                for b in range(pool_size):
                    value = X[i * stride + a][j * stride + b]
                    max_value = max(max_value, value)

            row.append(max_value)

        output.append(row)

    return output
