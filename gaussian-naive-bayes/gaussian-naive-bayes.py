import numpy as np

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns predicted class labels for X_test.
    """
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=float)

    classes = np.unique(y_train)
    predictions = []

    for x in X_test:
        best_class = None
        best_score = -np.inf

        for c in classes:
            X_c = X_train[y_train == c]
            prior = len(X_c) / len(X_train)

            mean = np.mean(X_c, axis=0)
            variance = np.var(X_c, axis=0) + 1e-9

            score = np.log(prior)
            score += np.sum(
                -0.5 * np.log(2 * np.pi * variance)
                - (x - mean) ** 2 / (2 * variance)
            )

            if score > best_score:
                best_score = score
                best_class = c

        predictions.append(int(best_class))

    return predictions