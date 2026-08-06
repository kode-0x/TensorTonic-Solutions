def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = float(x0)

    for _ in range(steps):
        x = x - lr * (2 * a * x + b)

    return x