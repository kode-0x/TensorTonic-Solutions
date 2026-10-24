def lbfgs_direction(grad: list, s_list: list, y_list: list) -> list:
    """
    Returns the L-BFGS descent direction from the stored history.
    """
    m = len(s_list)
    q = grad.copy()
    rho = []
    alpha = [0.0] * m

    for i in range(m):
        rho.append(1.0 / sum(a * b for a, b in zip(y_list[i], s_list[i])))

    for i in range(m - 1, -1, -1):
        alpha[i] = rho[i] * sum(a * b for a, b in zip(s_list[i], q))
        q = [a - alpha[i] * b for a, b in zip(q, y_list[i])]

    s = s_list[-1]
    y = y_list[-1]
    gamma = sum(a * b for a, b in zip(s, y)) / sum(a * a for a in y)
    r = [gamma * a for a in q]

    for i in range(m):
        beta = rho[i] * sum(a * b for a, b in zip(y_list[i], r))
        r = [a + b * (alpha[i] - beta) for a, b in zip(r, s_list[i])]

    return [-a for a in r]
