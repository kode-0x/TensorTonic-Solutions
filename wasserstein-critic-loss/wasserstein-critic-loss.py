import numpy as np

def wasserstein_critic_loss(real_scores: list, fake_scores: list) -> float:
    """
    Returns the loss as a float.
    """
    real_scores = np.asarray(real_scores)
    fake_scores = np.asarray(fake_scores)

    fake_scores_count = len(fake_scores)
    real_scores_count = len(real_scores)

    return float(np.mean(fake_scores) - np.mean(real_scores))