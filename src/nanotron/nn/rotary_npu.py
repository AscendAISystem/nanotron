import torch
import torch.nn as nn


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device) / dim))
    t = torch.arange(max_seq_len, dtype=dtype, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(freqs_cis)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    inplace: bool = False,
) -> torch.Tensor:
    if interleaved:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.cat((-x2, x1), dim=-1)
    else:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)

    result = x * cos + rotated * sin
    return result


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
        interleaved: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.interleaved = interleaved

        freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        batch_size, seq_length, num_heads, inner_dim = x.shape
        dtype = x.dtype

        assert inner_dim % 2 == 0
        orig_dtype = x.dtype
        x = x.to(torch.float32).view(batch_size, seq_length, num_heads, inner_dim // 2, 2)
        complex_x = torch.view_as_complex(x)

        if position_ids is None:
            freqs_cis = self.freqs_cis[None, :seq_length, None, :]
        else:
            freqs_cis = self.freqs_cis[position_ids][:, :, None, :]
        freqs_cis = freqs_cis.to(torch.float32)

        complex_freqs = torch.view_as_complex(freqs_cis)
        x_out = torch.view_as_real(complex_x * complex_freqs).view(
            batch_size, seq_length, num_heads, inner_dim
        )
        return x_out.to(orig_dtype)


__all__ = [
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "RotaryEmbedding",
]
