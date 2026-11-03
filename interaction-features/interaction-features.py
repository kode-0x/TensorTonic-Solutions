def interaction_features(X: list) -> list:
    result = []

    for row in X:
        expanded_row = row.copy()
        d = len(row)

        for i in range(d):
            for j in range(i + 1, d):
                expanded_row.append(row[i] * row[j])

        result.append(expanded_row)

    return result