import math

def gaussian_kernel(size: int, sigma: float) -> list:
    center = size // 2
    kernel = []

    for i in range(size):
        row = []
        for j in range(size):
            x = i - center
            y = j - center
            value = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
            row.append(value)
        kernel.append(row)

    total = sum(sum(row) for row in kernel)

    return [[value / total for value in row] for row in kernel]