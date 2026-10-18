def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    """
    sequences = [list(s.encode("utf-8")) for s in corpus]
    token_bytes = {i: bytes([i]) for i in range(256)}
    vocab = []
    merges = []
    next_id = 256

    while next_id < vocab_size:
        counts = {}

        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                counts[pair] = counts.get(pair, 0) + 1

        if not counts:
            break

        best_pair = max(
            counts,
            key=lambda p: (
                counts[p],
                token_bytes[p[0]],
                token_bytes[p[1]]
            )
        )

        left, right = best_pair
        merged_bytes = token_bytes[left] + token_bytes[right]

        vocab.append([next_id, list(merged_bytes)])
        merges.append([left, right, next_id])
        token_bytes[next_id] = merged_bytes

        for i, seq in enumerate(sequences):
            result = []
            j = 0

            while j < len(seq):
                if (
                    j + 1 < len(seq)
                    and seq[j] == left
                    and seq[j + 1] == right
                ):
                    result.append(next_id)
                    j += 2
                else:
                    result.append(seq[j])
                    j += 1

            sequences[i] = result

        next_id += 1

    return {"vocab": vocab, "merges": merges}
