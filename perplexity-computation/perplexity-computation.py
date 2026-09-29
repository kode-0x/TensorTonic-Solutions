import math

def perplexity(prob_distributions: list, actual_tokens: list) -> float:
    """
    Returns the sequence perplexity.
    """
    return round(
        math.exp(-sum(math.log(p[t]) for p, t in zip(prob_distributions, actual_tokens)) / len(actual_tokens)), 4)
