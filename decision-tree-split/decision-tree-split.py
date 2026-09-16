def decision_tree_split(X: list, y: list) -> list:
    """
    Returns the best feature index and threshold.
    """
    n = len(y)
    best_gain = -1
    best_feature = 0
    best_threshold = 0.0

    def gini(labels):
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        p = counts / len(labels)
        return 1.0 - np.sum(p ** 2)

    parent_gini = gini(y)

    for feature in range(len(X[0])):
        values = np.unique([row[feature] for row in X])

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2

            left = [y[j] for j in range(n) if X[j][feature] <= threshold]
            right = [y[j] for j in range(n) if X[j][feature] > threshold]

            weighted_gini = (
                len(left) / n * gini(left)
                + len(right) / n * gini(right)
            )

            gain = parent_gini - weighted_gini

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return [best_feature, best_threshold]