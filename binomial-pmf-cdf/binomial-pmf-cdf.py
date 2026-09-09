import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    pmf = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    cdf = 0.0
    for i in range(k + 1):
        cdf += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    return {
        "pmf": float(pmf),
        "cdf": float(cdf)
    }