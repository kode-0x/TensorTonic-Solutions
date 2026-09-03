import numpy as np

def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    values = np.asarray(values, dtype=float)
    transitions = np.asarray(transitions, dtype=float)
    rewards = np.asarray(rewards, dtype=float)

    q_values = rewards + gamma * np.sum(transitions * values[None, None, :], axis=2)

    return np.max(q_values, axis=1).tolist()