import numpy as np

def bert_embeddings(token_ids: np.ndarray, segment_ids: np.ndarray,
                    token_embeddings: np.ndarray, position_embeddings: np.ndarray,
                    segment_embeddings: np.ndarray) -> np.ndarray:
    """
    Returns the float64 BERT input embeddings with shape (B, S, H).
    """
    token_ids = np.asarray(token_ids, dtype=np.int64)
    segment_ids = np.asarray(segment_ids, dtype=np.int64)

    B, S = token_ids.shape
    positions = np.arange(S)

    token_emb = token_embeddings[token_ids]
    position_emb = position_embeddings[positions][None, :, :]
    segment_emb = segment_embeddings[segment_ids]

    return np.asarray(token_emb + position_emb + segment_emb, dtype=np.float64)