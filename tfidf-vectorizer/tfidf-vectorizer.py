import math
from collections import Counter
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    """
    Returns a dictionary with tfidf_matrix and vocabulary.
    """
    tokens = [doc.lower().split() for doc in documents]
    vocabulary = sorted(set(word for doc in tokens for word in doc))

    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    N = len(documents)
    V = len(vocabulary)

    tfidf_matrix = np.zeros((N, V), dtype=float)

    df = Counter()
    for doc in tokens:
        df.update(set(doc))

    idf = np.array([
        math.log(N / df[word])
        for word in vocabulary
    ])

    for i, doc in enumerate(tokens):
        counts = Counter(doc)
        length = len(doc)

        for word, count in counts.items():
            j = word_to_idx[word]
            tfidf_matrix[i, j] = (count / length) * idf[j]

    return {
        "tfidf_matrix": tfidf_matrix,
        "vocabulary": vocabulary
    }