import math

def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    result = []

    for value in x:
        if value > 0:
            result.append(float(value))
        else:
            result.append(float(alpha * (math.exp(value) - 1)))

    return result