import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor,
                  center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> dict:
    """
    Returns updated W_in and W_out float64 tensors in a dictionary.
    """
    v = W_in[center_id].clone()
    u_pos = W_out[pos_id].clone()
    u_neg = W_out[neg_ids].clone()

    pos_score = torch.dot(v, u_pos)
    neg_scores = u_neg @ v

    pos_sig = torch.sigmoid(pos_score)
    neg_sig = torch.sigmoid(neg_scores)

    grad_u_pos = (pos_sig - 1.0) * v
    grad_u_neg = neg_sig[:, None] * v
    grad_v = (pos_sig - 1.0) * u_pos + (neg_sig[:, None] * u_neg).sum(dim=0)

    new_W_in = W_in.clone()
    new_W_out = W_out.clone()

    new_W_in[center_id] -= lr * grad_v

    new_W_out[pos_id] -= lr * grad_u_pos

    for i, neg_id in enumerate(neg_ids):
        new_W_out[neg_id] -= lr * grad_u_neg[i]

    return {"W_in": new_W_in, "W_out": new_W_out}
