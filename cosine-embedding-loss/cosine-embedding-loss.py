import math

def cosine_embedding_loss(x1: list, x2: list, label: int, margin: float) -> float:
    """
    Returns the cosine embedding loss as a float.
    """
    dot = sum(a * b for a, b in zip(x1, x2))

    norm_x1 = math.sqrt(sum(a * a for a in x1))
    norm_x2 = math.sqrt(sum(b * b for b in x2))

    cosine_similarity = dot / (norm_x1 * norm_x2)

    if label == 1:
        loss = 1 - cosine_similarity
    else:
        loss = max(0, cosine_similarity - margin)

    return float(loss)