import numpy as np

def apply_mlm_mask(token_ids: np.ndarray, mask_positions: np.ndarray,
                   replace_probs: np.ndarray, random_tokens: np.ndarray,
                   mask_token_id: int = 103) -> dict:
    """
    Returns masked_ids and labels as int64 arrays in a dictionary.
    """
    masked_ids = np.array(token_ids, dtype=np.int64, copy=True)
    labels = np.full(token_ids.shape, -100, dtype=np.int64)

    selected = mask_positions
    labels[selected] = token_ids[selected]

    use_mask = selected & (replace_probs < 0.8)
    use_random = selected & (replace_probs >= 0.8) & (replace_probs < 0.9)

    masked_ids[use_mask] = mask_token_id
    masked_ids[use_random] = random_tokens[use_random]

    return {
        "masked_ids": masked_ids,
        "labels": labels
    }