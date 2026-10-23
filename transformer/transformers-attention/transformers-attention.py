import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32, device=K.device))
    attention_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, V)