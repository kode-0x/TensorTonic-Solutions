import math
from collections import Counter

def bleu_score(candidate: list, reference: list, max_n: int) -> float:
    """
    Returns the unsmoothed BLEU score.
    """
    if not candidate:
        return 0.0

    precisions = []

    for n in range(1, max_n + 1):
        candidate_ngrams = Counter(
            tuple(candidate[i:i + n])
            for i in range(len(candidate) - n + 1)
        )
        reference_ngrams = Counter(
            tuple(reference[i:i + n])
            for i in range(len(reference) - n + 1)
        )

        total = sum(candidate_ngrams.values())

        if total == 0:
            return 0.0

        clipped = sum(
            min(count, reference_ngrams[ngram])
            for ngram, count in candidate_ngrams.items()
        )

        precision = clipped / total

        if precision == 0:
            return 0.0

        precisions.append(precision)

    c = len(candidate)
    r = len(reference)

    bp = 1.0 if c >= r else math.exp(1 - r / c)

    return float(bp * math.exp(sum(math.log(p) for p in precisions) / max_n))