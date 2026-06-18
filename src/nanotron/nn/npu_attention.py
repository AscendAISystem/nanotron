import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from nanotron.npu_compat import is_npu_available
import torch_npu


def _npu_flash_attention_prompt(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    atten_mask: Optional[torch.Tensor] = None,
    actual_seq_lengths: Optional[torch.Tensor] = None,
    num_heads: int = 1,
    num_key_value_heads: int = 0,
    input_layout: str = "BNSD",
) -> torch.Tensor:
    sparse_mode = 0
    if atten_mask is not None:
        sparse_mode = 1
    out, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
        q.contiguous(), k.contiguous(), v.contiguous(),
        head_num=num_heads,
        input_layout=input_layout,
        scale=scale if scale is not None else 1.0 / math.sqrt(q.size(-1)),
        keep_prob=1.0,
        atten_mask=atten_mask.contiguous() if atten_mask is not None else None,
        sparse_mode=sparse_mode,
    )
    return out


def _npu_flash_attention_incre(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    atten_mask: Optional[torch.Tensor] = None,
    actual_seq_lengths: Optional[torch.Tensor] = None,
    num_heads: int = 1,
    num_key_value_heads: int = 0,
    input_layout: str = "BSH",
) -> torch.Tensor:
    kwargs = dict(
        query=q.contiguous(),
        key=k.contiguous(),
        value=v.contiguous(),
        num_heads=num_heads,
        input_layout=input_layout,
        scale_value=scale if scale is not None else 1.0 / math.sqrt(q.size(-1)),
        num_key_value_heads=num_key_value_heads if num_key_value_heads > 0 else num_heads,
    )
    if atten_mask is not None:
        kwargs["atten_mask"] = atten_mask.contiguous()
    if actual_seq_lengths is not None:
        actual_seq_lengths = actual_seq_lengths.int().tolist()
        if isinstance(actual_seq_lengths, list):
            kwargs["actual_seq_lengths"] = actual_seq_lengths
    return torch_npu.npu_incre_flash_attention(**kwargs)


def npu_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    if not is_npu_available():
        raise RuntimeError("npu_flash_attention requires an Ascend NPU device.")

    B, H, S, D = q.shape
    try:
        if is_causal or attn_mask is not None:
            mask = None
            if is_causal:
                mask = torch.triu(torch.full((S, S), 1, dtype=torch.uint8, device=q.device), diagonal=1)
            if attn_mask is not None:
                mask = attn_mask if mask is None else (mask + attn_mask > 0).to(torch.uint8)

            output = _npu_flash_attention_prompt(
                q.transpose(1, 2).contiguous() if q.shape[1] == H else q.contiguous(),
                k.transpose(1, 2).contiguous() if k.shape[1] == H else k.contiguous(),
                v.transpose(1, 2).contiguous() if v.shape[1] == H else v.contiguous(),
                scale=scale,
                atten_mask=mask,
                num_heads=H,
                num_key_value_heads=H,
                input_layout="BSH" if q.shape[1] == S else "BNSD",
            )
        else:
            output = _npu_flash_attention_prompt(
                q.transpose(1, 2).contiguous() if q.shape[1] == H else q.contiguous(),
                k.transpose(1, 2).contiguous() if k.shape[1] == H else k.contiguous(),
                v.transpose(1, 2).contiguous() if v.shape[1] == H else v.contiguous(),
                scale=scale,
                num_heads=H,
                num_key_value_heads=H,
                input_layout="BSH" if q.shape[1] == S else "BNSD",
            )
    except Exception as e:
        raise RuntimeError(f"NPU flash attention failed: {e}") from e

    if output.dim() == 4 and output.shape[1] != H:
        output = output.transpose(1, 2).contiguous()
    return output


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    if attn_mask is not None and attn_mask.dtype not in (torch.bool, q.dtype):
        attn_mask = attn_mask.to(q.dtype)
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def _run_with_fallback(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    if is_npu_available() and q.dtype in (torch.float16, torch.bfloat16):
        try:
            return npu_flash_attention(q, k, v, scale=scale, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        except Exception:
            pass
    return sdpa_attention(q, k, v, scale=scale, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)


def npu_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    return_attn_probs: bool = False,
):
    out, _, _, _, _, _ = npu_fusion_attn_varlen_fwd(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p=dropout_p, softmax_scale=softmax_scale,
        causal=causal, window_size=window_size,
    )
    if return_attn_probs:
        return out, None, None
    return out


def npu_flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    rotary_interleaved: bool = False,
    **kwargs,
) -> torch.Tensor:
    batch_size = q.shape[0]
    device = q.device

    if cache_seqlens is not None:
        for i in range(batch_size):
            seq_len = int(cache_seqlens[i].item())
            k_len = k.shape[1] if k.dim() >= 2 else 1
            k_cache[i, seq_len:seq_len + k_len] = k[i]
            v_cache[i, seq_len:seq_len + k_len] = v[i]

    outputs = []

    for i in range(batch_size):
        if cache_seqlens is not None:
            seq_len = int(cache_seqlens[i].item())
            total_kv_len = seq_len + (k.shape[1] if k.dim() >= 2 else 1)
            k_i = k_cache[i, :total_kv_len].unsqueeze(0)
            v_i = v_cache[i, :total_kv_len].unsqueeze(0)
        else:
            k_i = torch.cat([k_cache[i], k[i]], dim=0).unsqueeze(0)
            v_i = torch.cat([v_cache[i], v[i]], dim=0).unsqueeze(0)

        q_i = q[i:i + 1]

        if q_i.dim() == 4:
            q_i = q_i.transpose(1, 2).contiguous()
            k_i = k_i.transpose(1, 2).contiguous()
            v_i = v_i.transpose(1, 2).contiguous()

        out_i = _run_with_fallback(
            q_i, k_i, v_i,
            scale=softmax_scale,
            dropout_p=0.0,
            is_causal=causal,
        )

        if out_i.dim() == 4:
            out_i = out_i.transpose(1, 2).contiguous()
        outputs.append(out_i)

    return torch.cat(outputs, dim=0)


def npu_flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    **kwargs,
):
    # kv shape: [T, 2, kv_heads, head_dim]; split into k/v via dim=1
    k, v = kv[:, 0], kv[:, 1]

    # Handle GQA: expand k/v heads to match q heads
    n_heads = q.size(1)
    n_kv_heads = k.size(1)
    if n_kv_heads < n_heads:
        assert n_heads % n_kv_heads == 0, f"q_heads({n_heads}) not divisible by kv_heads({n_kv_heads})"
        repeat = n_heads // n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    return_attn_probs = kwargs.pop("return_attn_probs", False)

    return npu_flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=return_attn_probs,
    )


def npu_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    return_attn_probs: bool = False,
):
    return _run_with_fallback(
        q, k, v,
        scale=softmax_scale,
        dropout_p=dropout_p,
        is_causal=causal,
    )


def unpad_input(
    tensor: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    seqlen = tensor.shape[1]
    indices = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        return tensor.new_zeros((0, *tensor.shape[2:])), indices.new_zeros(0), indices.new_zeros(2), 0

    unpad = tensor.flatten(0, 1)[indices]
    cu_seqlens = torch.zeros(2, dtype=torch.int32, device=indices.device)
    cu_seqlens[1] = indices.numel()
    max_seqlen = seqlen
    return unpad, indices, cu_seqlens, max_seqlen


def pad_input(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    seqlen: int,
) -> torch.Tensor:
    output = tensor.new_zeros(batch_size, seqlen, *tensor.shape[1:])
    if tensor.numel() == 0:
        return output
    output.flatten(0, 1)[indices] = tensor
    return output


def npu_fusion_attn_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
):
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.size(-1))

    actual_seq_qlen = cu_seqlens_q[1:].tolist()
    actual_seq_kvlen = cu_seqlens_k[1:].tolist()

    sparse_mode = 3 if causal else 0
    pre_tockens = window_size[0] if window_size[0] > 0 else max_seqlen_q
    next_tockens = window_size[1] if window_size[1] > 0 else max_seqlen_k

    out, softmax_max, softmax_sum, _, seed, offset, numels = torch_npu.npu_fusion_attention(
        q.contiguous(), k.contiguous(), v.contiguous(),
        head_num=q.size(1),
        input_layout="TND",
        scale=softmax_scale,
        keep_prob=1.0 - dropout_p,
        pre_tockens=pre_tockens,
        next_tockens=next_tockens,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=sparse_mode,
    )
    return out, softmax_max, softmax_sum, seed, offset, numels


def npu_fusion_attn_varlen_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    seed: int = 0,
    offset: int = 0,
    numels: int = 0,
):
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.size(-1))

    actual_seq_qlen = cu_seqlens_q[1:].tolist()
    actual_seq_kvlen = cu_seqlens_k[1:].tolist()

    sparse_mode = 3 if causal else 0
    pre_tockens = window_size[0] if window_size[0] > 0 else max_seqlen_q
    next_tockens = window_size[1] if window_size[1] > 0 else max_seqlen_k

    dq, dk, dv, _, _ = torch_npu.npu_fusion_attention_grad(
        q.contiguous(), k.contiguous(), v.contiguous(), dout.contiguous(),
        head_num=q.size(1),
        input_layout="TND",
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=out.contiguous(),
        scale_value=softmax_scale,
        keep_prob=1.0 - dropout_p,
        pre_tockens=pre_tockens,
        next_tockens=next_tockens,
        seed=seed,
        offset=offset,
        numels=numels,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=sparse_mode,
    )
    return dq, dk, dv


def npu_fusion_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
):
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.size(-1))

    sparse_mode = 0
    pre_tockens = 2147483647
    next_tockens = 2147483647

    if causal and window_size == (-1, -1):
        sparse_mode = 3
    elif window_size != (-1, -1):
        sparse_mode = 4
        pre_tockens = window_size[0]
        next_tockens = window_size[1]

    out, softmax_max, softmax_sum, _, seed, offset, numels = torch_npu.npu_fusion_attention(
        q.contiguous(), k.contiguous(), v.contiguous(),
        head_num=q.size(1),
        input_layout="BNSD" if q.dim() == 4 else "BSH",
        scale=softmax_scale,
        keep_prob=1.0 - dropout_p,
        pre_tockens=pre_tockens,
        next_tockens=next_tockens,
        sparse_mode=sparse_mode,
    )
    return out, softmax_max, softmax_sum, seed, offset, numels


def npu_fusion_attn_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    seed: int = 0,
    offset: int = 0,
    numels: int = 0,
):
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.size(-1))

    sparse_mode = 0
    pre_tockens = 2147483647
    next_tockens = 2147483647

    if causal and window_size == (-1, -1):
        sparse_mode = 3
    elif window_size != (-1, -1):
        sparse_mode = 4
        pre_tockens = window_size[0]
        next_tockens = window_size[1]

    dq, dk, dv, _, _ = torch_npu.npu_fusion_attention_grad(
        q.contiguous(), k.contiguous(), v.contiguous(), dout.contiguous(),
        head_num=q.size(1),
        input_layout="BNSD" if q.dim() == 4 else "BSH",
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=out.contiguous(),
        scale_value=softmax_scale,
        keep_prob=1.0 - dropout_p,
        pre_tockens=pre_tockens,
        next_tockens=next_tockens,
        seed=seed,
        offset=offset,
        numels=numels,
        sparse_mode=sparse_mode,
    )
    return dq, dk, dv


__all__ = [
    "npu_flash_attention",
    "sdpa_attention",
    "npu_flash_attn_varlen_func",
    "npu_flash_attn_with_kvcache",
    "npu_flash_attn_varlen_kvpacked_func",
    "npu_flash_attn_func",
    "unpad_input",
    "pad_input",
    "npu_fusion_attn_varlen_fwd",
    "npu_fusion_attn_varlen_bwd",
    "npu_fusion_attn_fwd",
    "npu_fusion_attn_bwd",
]
