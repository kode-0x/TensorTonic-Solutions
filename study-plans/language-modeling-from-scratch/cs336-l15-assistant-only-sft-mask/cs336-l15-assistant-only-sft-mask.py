import torch

def assistant_only_sft_mask(
    input_ids: torch.Tensor, role_ids: torch.Tensor,
    attention_mask: torch.Tensor, assistant_role: int = 1,
) -> dict:
    """
    Returns a dict: labels (input token dtype), loss_mask (Boolean tensor).
    """
    labels = torch.full_like(input_ids, -100)
    loss_mask = torch.zeros_like(attention_mask, dtype=torch.bool)

    if input_ids.shape[1] == 0:
        return {
            "labels": labels,
            "loss_mask": loss_mask,
        }

    # Position t predicts the token at position t + 1.
    target_is_assistant = role_ids[:, 1:] == assistant_role
    transition_attended = (
        attention_mask[:, :-1] & attention_mask[:, 1:]
    )

    mask = target_is_assistant & transition_attended

    labels[:, :-1] = torch.where(
        mask,
        input_ids[:, 1:],
        labels[:, :-1],
    )
    loss_mask[:, :-1] = mask

    return {
        "labels": labels,
        "loss_mask": loss_mask,
    }