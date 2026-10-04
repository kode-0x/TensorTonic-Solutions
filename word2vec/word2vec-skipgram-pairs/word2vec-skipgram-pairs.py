import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns the ordered center-context pairs as an int64 tensor.
    """
    n = token_ids.numel()
    pairs = []

    for i in range(n):
        start = max(0, i - window)
        end = min(n - 1, i + window)

        for j in range(start, end + 1):
            if j != i:
                pairs.append((token_ids[i].item(), token_ids[j].item()))

    if not pairs:
        return torch.empty((0, 2), dtype=torch.int64)

    return torch.tensor(pairs, dtype=torch.int64)
