import numpy as np

def bigram_probabilities(tokens: list) -> dict:
    """
    Returns a dictionary with vocab, counts, and probabilities.
    """
    vocab = sorted(set(tokens))
    V = len(vocab)
    index = {token: i for i, token in enumerate(vocab)}

    counts = np.zeros((V, V), dtype=int)

    for a, b in zip(tokens, tokens[1:]):
        counts[index[a], index[b]] += 1

    probabilities = (counts + 1) / (counts.sum(axis=1, keepdims=True) + V)

    return {
        "vocab": vocab,
        "counts": counts,
        "probabilities": probabilities
    }