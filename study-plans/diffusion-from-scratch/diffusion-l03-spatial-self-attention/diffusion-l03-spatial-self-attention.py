import math
import torch
import torch.nn.functional as F

def spatial_self_attention(x: torch.Tensor, params: dict, num_groups: int, num_heads: int) -> torch.Tensor:
    """
    Returns x plus the attention output, shape (B, C, H, W) with the dtype of x.
    """
    B, C, H, W = x.shape
    d = C // num_heads

    z = F.group_norm(
        x,
        num_groups,
        params["norm_weight"],
        params["norm_bias"],
        eps=1e-5
    )

    z = z.permute(0, 2, 3, 1).reshape(B, H * W, C)

    q = F.linear(z, params["q_weight"], params["q_bias"])
    k = F.linear(z, params["k_weight"], params["k_bias"])
    v = F.linear(z, params["v_weight"], params["v_bias"])

    q = q.reshape(B, H * W, num_heads, d).transpose(1, 2)
    k = k.reshape(B, H * W, num_heads, d).transpose(1, 2)
    v = v.reshape(B, H * W, num_heads, d).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    attention = torch.softmax(scores, dim=-1)
    out = torch.matmul(attention, v)

    out = out.transpose(1, 2).reshape(B, H * W, C)

    out = F.linear(out, params["out_weight"], params["out_bias"])

    out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

    return x + out