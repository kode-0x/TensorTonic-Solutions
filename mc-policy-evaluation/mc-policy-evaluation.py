import numpy as np

def mc_policy_evaluation(episodes: list, gamma: float, n_states: int) -> np.ndarray:
    returns = [[] for _ in range(n_states)]

    for episode in episodes:
        G = 0.0

        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]
            G = reward + gamma * G

            if state not in [s for s, _ in episode[:t]]:
                returns[state].append(G)

    values = np.zeros(n_states)

    for state in range(n_states):
        if returns[state]:
            values[state] = np.mean(returns[state])

    return np.round(values, 4)