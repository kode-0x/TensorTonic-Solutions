import torch

def kv_cache_decode_step(
    query: torch.Tensor, new_key: torch.Tensor, new_value: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor,
    num_query_heads: int, num_kv_heads: int) -> dict:
    """
    Returns a dict of tensors: output, new_key_cache, new_value_cache.
    """
    new_key_cache = torch.cat((key_cache, new_key.unsqueeze(2)), dim=2)
    new_value_cache = torch.cat((value_cache, new_value.unsqueeze(2)), dim=2)

    B, Hq, D = query.shape
    G = num_query_heads // num_kv_heads
    S = new_key_cache.shape[2]

    q = query.reshape(B, num_kv_heads, G, D).float()
    k = new_key_cache.float()
    v = new_value_cache.float()

    scores = torch.einsum("bhgd,bhsd->bhgs", q, k) / (D ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhgs,bhsd->bhgd", weights, v)

    return {
        "output": output.reshape(B, Hq, D).to(query.dtype),
        "new_key_cache": new_key_cache,
        "new_value_cache": new_value_cache
    }