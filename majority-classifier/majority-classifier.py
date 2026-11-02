import numpy as np

def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a one-dimensional NumPy array.
    """
    labels, counts = np.unique(y_train, return_counts=True)

    majority_label = labels[np.argmax(counts)]

    for label in y_train:
        if y_train.count(label) == max(counts):
            majority_label = label
            break

    return np.full(len(X_test), majority_label, dtype=int)