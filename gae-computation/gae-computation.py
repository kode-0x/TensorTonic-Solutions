def gae(rewards: list, values: list, gamma: float, lam: float) -> list:
    """
    Returns the generalized advantage estimate at every timestep.
    """
    advantages = [0.0] * len(rewards)
    next_advantage = 0.0

    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        next_advantage = delta + gamma * lam * next_advantage
        advantages[t] = next_advantage

    return advantages