import math
from collections import Counter
import numpy as np

def bm25_score(query_tokens: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    if not docs:
        return np.array([], dtype=float)

    N = len(docs)
    avgdl = sum(len(doc) for doc in docs) / N

    df = Counter()
    for doc in docs:
        df.update(set(doc))

    scores = []

    for doc in docs:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0

        for term in set(query_tokens):
            if term not in tf:
                continue

            idf = math.log(
                (N - df[term] + 0.5) / (df[term] + 0.5) + 1
            )

            denom = tf[term] + k1 * (
                1 - b + b * dl / avgdl
            )

            score += idf * (
                tf[term] * (k1 + 1)
            ) / denom

        scores.append(score)

    return np.array(scores, dtype=float)