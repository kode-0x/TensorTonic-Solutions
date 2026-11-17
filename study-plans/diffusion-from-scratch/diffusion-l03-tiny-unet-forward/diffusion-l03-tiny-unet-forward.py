import math
import torch
import torch.nn.functional as F

def tiny_unet_forward(x: torch.Tensor, timesteps: torch.Tensor, params: dict, num_groups: int) -> torch.Tensor:
    """
    Returns the network output of shape (B, C_out, H, W) with the dtype of x.
    """
    dtype = x.dtype
    B, _, H, W = x.shape

    def conv(x, prefix, stride=1):
        return F.conv2d(
            x,
            params[prefix + "_weight"],
            params[prefix + "_bias"],
            stride=stride,
            padding=1
        )

    def norm(x, prefix):
        return F.group_norm(
            x,
            num_groups,
            params[prefix + "_weight"],
            params[prefix + "_bias"],
            eps=1e-5
        )

    def residual(x, e, prefix):
        h = norm(x, prefix + "_norm1")
        h = F.silu(h)
        h = F.conv2d(
            h,
            params[prefix + "_conv1_weight"],
            params[prefix + "_conv1_bias"],
            padding=1
        )

        time = F.linear(
            F.silu(e),
            params[prefix + "_time_weight"],
            params[prefix + "_time_bias"]
        )
        h = h + time[:, :, None, None]

        h = norm(h, prefix + "_norm2")
        h = F.silu(h)
        h = F.conv2d(
            h,
            params[prefix + "_conv2_weight"],
            params[prefix + "_conv2_bias"],
            padding=1
        )

        if prefix + "_skip_weight" in params:
            skip = F.conv2d(
                x,
                params[prefix + "_skip_weight"],
                params[prefix + "_skip_bias"]
            )
        else:
            skip = x

        return h + skip

    half = torch.arange(E := params["time_linear1_weight"].shape[1] // 2, device=x.device, dtype=dtype)
    frequencies = 10000.0 ** (-2.0 * half / (2 * E))
    angles = timesteps.to(dtype).unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)

    e = F.linear(
        embedding,
        params["time_linear1_weight"],
        params["time_linear1_bias"]
    )
    e = F.silu(e)
    e = F.linear(
        e,
        params["time_linear2_weight"],
        params["time_linear2_bias"]
    )

    h1 = conv(x, "conv_in")
    h1 = residual(h1, e, "down_res")

    h = conv(h1, "downsample", stride=2)
    h = residual(h, e, "mid_res")

    h = h.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    h = conv(h, "upsample")

    h = torch.cat([h, h1], dim=1)
    h = residual(h, e, "up_res")

    h = norm(h, "out_norm")
    h = F.silu(h)
    out = conv(h, "conv_out")

    return out.to(dtype)