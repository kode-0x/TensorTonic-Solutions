import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    index = {word: i for i, word in enumerate(vocab)}
    result = np.zeros(len(vocab), dtype=int)

    for token in tokens:
        if token in index:
            result[index[token]] += 1

    return result
