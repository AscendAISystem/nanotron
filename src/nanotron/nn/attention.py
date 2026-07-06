from functools import lru_cache
from typing import Literal, Optional, Tuple

import torch
from packaging import version

from nanotron.npu_utils import is_npu_available


# Lazy imports for ring/custom attention (flash_attn dependency)
def get_ring_flash_attn_varlen_func():
    from nanotron.nn.ring_attention import ring_flash_attn_varlen_func

    return ring_flash_attn_varlen_func


def get_llama3_flash_attn_varlen_qkvpacked_func():
    from nanotron.nn.llama3_ring_attention import llama3_flash_attn_varlen_qkvpacked_func

    return llama3_flash_attn_varlen_qkvpacked_func


# Replace direct import with a function for lazy loading
def get_ring_flash_attn_cuda():
    """Lazily import ring_flash_attn_cuda to avoid early Triton dependency."""
    from nanotron.nn.ring_attention_lucidrain import ring_flash_attn_cuda

    return ring_flash_attn_cuda


@lru_cache()
def is_torch_flex_attn_available():
    # TODO check if some bugs cause push backs on the exact version
    # NOTE: We require torch>=2.5.0 as it is the first release
    return version.parse(torch.__version__) >= version.parse("2.5.0")


# flex_attention requires torch>=2.5 and is not available on NPU
if is_torch_flex_attn_available() and not is_npu_available():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention


@lru_cache()
def is_flash_attn_greater_or_equal_2_10():
    try:
        import flash_attn

        return version.parse(flash_attn.__version__) >= version.parse("2.1.0")
    except ImportError:
        return False


# flash_attn is not available on NPU
if not is_npu_available() and is_flash_attn_greater_or_equal_2_10():
    from flash_attn.flash_attn_interface import flash_attn_func
# adapted from transformers.integrations.flex_attention.flex_attention_forward
def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    sliding_window: Optional[int] = None,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    position_ids: Optional[torch.Tensor] = None,
    document_ids: Optional[torch.Tensor] = None,
    flex_attention_mask: Optional[str] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implementation of attention using PyTorch's FlexAttention.

    Args:
        module: The module calling this function
        query: Query states tensor [batch_size, num_heads, seq_len, head_dim]
        key: Key states tensor [batch_size, num_kv_heads, seq_len, head_dim]
        value: Value states tensor [batch_size, num_kv_heads, seq_len, head_dim]
        attention_mask: Optional attention mask tensor
        sliding_window: Optional sliding window size for efficient attention
        scaling: Optional scaling factor for attention scores
        softcap: Optional softcap for softmax stability
        position_ids: Optional tensor of position IDs used for document masking
        document_ids: Optional tensor explicitly marking document boundaries [seq_len]
                     (e.g., [0,0,0,1,1,2,2,2,2,2,2] for seqs of length 3,2,6)
        flex_attention_mask: Optional string specifying a custom mask type
        
    Returns:
        Tuple of (attention_output, attention_weights)
    """
    from nanotron.nn.flex_attention import (
        create_softcapped_causal_score_mod,
        create_document_mask_func,
        create_attention_mask,
        get_attention_mod_from_type,
        get_block_mask_from_type,
        validate_attention_args,
    )

    # Validate arguments if a flex_attention_mask is specified
    validate_attention_args(
        flex_attention_mask=flex_attention_mask,
        sliding_window=sliding_window,
        position_ids=position_ids,
        document_ids=document_ids,
    )

    # We're setting score_mod to None as requested
    score_mod = None

    # Determine which block mask to use
    if flex_attention_mask is not None:
        # Use the mask type specified by flex_attention_mask
        block_mask = get_block_mask_from_type(
            flex_attention_mask=flex_attention_mask,
            query=query,
            key=key,
            sliding_window=sliding_window,
            position_ids=position_ids,
            document_ids=document_ids,
        )
    else:
        # Use the existing document/sliding window masking logic
        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]
            
        # Create document masking function if needed
        doc_mask_func = create_document_mask_func(query, document_ids, position_ids)

        # Create combined attention mask
        block_mask = create_attention_mask(query, sliding_window, doc_mask_func)

    # Call PyTorch's flex_attention with the appropriate parameters
    attn_output, attention_weights = flex_attention(
        query,
        key,
        value,
        enable_gqa=True,  # Enable grouped query attention
        score_mod=score_mod,
        block_mask=block_mask,  # Efficient mask based on type
        scale=scaling,
        return_lse=True,  # FlexAttention always computes log-sum-exp anyway
    )

    # FlexAttention returns weights in float32, convert to match value dtype
    attention_weights = attention_weights.to(value.dtype)

    # Transpose output to match expected format [batch_size, seq_len, num_heads, head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attention_weights


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,  # [b, num_heads, seq_len, head_dim]
    key: torch.Tensor,  # [b, num_kv_heads, seq_len, head_dim]
    value: torch.Tensor,  # [b, num_kv_heads, seq_len, head_dim]
    attention_mask: Optional[torch.Tensor],  # [b, num_heads, seq_len, seq_len]
    max_seqlen: Optional[int],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    query = query.view(-1, max_seqlen, module.local_num_heads, module.head_dim)
    key = key.view(-1, max_seqlen, module.local_num_kv_heads, module.head_dim)
    value = value.view(-1, max_seqlen, module.local_num_kv_heads, module.head_dim)

    if attention_mask is None:
        is_causal = True
    else:
        is_causal = False

    if sliding_window is not None:
        window_size = (sliding_window, sliding_window)
    else:
        window_size = (-1, -1)

    if is_npu_available():
        # NPU branch: use npu_fusion_attention if available, otherwise fallback to SDPA
        import torch_npu  # noqa: F401

        try:
            # npu_fusion_attention signature:
            #   query, key, value, head_num, input_layout, pse, padding_mask, softmax_scale, keep_prob, pre_tockens, next_tockens, attention_mask_type, scale, sparse_mode
            attn_output, _ = torch_npu.npu_fusion_attention(
                query,
                key,
                value,
                head_num=module.local_num_heads,
                input_layout="BSND",
                pse=None,
                padding_mask=None,
                softmax_scale=scaling,
                keep_prob=1.0 - dropout,
                pre_tockens=65536,
                next_tockens=65536,
                attention_mask_type="causal" if is_causal else "default",
                scale=1.0,
                sparse_mode=0,
            )
        except (RuntimeError, AttributeError) as e:
            # Fallback to SDPA if npu_fusion_attention fails
            import warnings
            warnings.warn(f"npu_fusion_attention failed ({e}), falling back to SDPA")

            # SDPA expects [b, num_heads, seq_len, head_dim]
            q_sdpa = query.transpose(1, 2).contiguous()
            k_sdpa = key.transpose(1, 2).contiguous()
            v_sdpa = value.transpose(1, 2).contiguous()
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attention_mask,
                dropout_p=dropout,
                scale=scaling,
                is_causal=is_causal,
                enable_gqa=query.shape[2] != key.shape[2],
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
    else:
        # CUDA/CPU: use flash_attn
        attn_output = flash_attn_func(
            q=query,
            k=key,
            v=value,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=is_causal,
            window_size=window_size,
            return_attn_probs=False,
        )
    attn_output = attn_output.contiguous()
    return attn_output, None


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,  # [b, num_heads, seq_len, head_dim]
    key: torch.Tensor,  # [b, num_kv_heads, seq_len, head_dim]
    value: torch.Tensor,  # [b, num_kv_heads, seq_len, head_dim]
    attention_mask: Optional[torch.Tensor],  # [b, num_heads, seq_len, seq_len]
    max_seqlen: int,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if attention_mask is None:
        is_causal = True
    else:
        is_causal = False
    query = query.view(-1, max_seqlen, module.local_num_heads, module.head_dim).transpose(
        1, 2
    )  # [b, num_heads, seq_length, head_dim]
    key = key.view(-1, max_seqlen, module.local_num_kv_heads, module.head_dim).transpose(1, 2)
    value = value.view(-1, max_seqlen, module.local_num_kv_heads, module.head_dim).transpose(1, 2)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    attn_output = attn_output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
    return attn_output, None


# On NPU, flash_attention_2 is not available, so we fall back to SDPA
if is_npu_available():
    _flash_attn_impl = sdpa_attention_forward
else:
    _flash_attn_impl = flash_attention_forward

ALL_ATTENTION_FUNCTIONS = {
    "flash_attention_2": _flash_attn_impl,
    "flex_attention": flex_attention_forward,
    "sdpa": sdpa_attention_forward,
}
# Register ring/triton attention with try/except guards for NPU compatibility
for _key, _getter, _name in [
    ("ring_flash_triton", get_ring_flash_attn_cuda, "ring_flash_attn_cuda"),
    ("ring", get_ring_flash_attn_varlen_func, "ring_flash_attn_varlen_func"),
    ("llama3_ring_attention", get_llama3_flash_attn_varlen_qkvpacked_func, "llama3_flash_attn_varlen_qkvpacked_func"),
]:
    try:
        # Resolve to verify import works
        _getter()
        ALL_ATTENTION_FUNCTIONS[_key] = lambda *a, g=_getter, **kw: g()(*a, **kw)
    except Exception:
        if is_npu_available():
            # NPU: register with clear error message
            def _make_npu_fallback(n=_name):
                def _fn(*a, **kw):
                    raise RuntimeError(
                        f"'{n}' requires Triton and flash_attn which are not available on NPU. "
                        "Use 'sdpa' or 'flash_attention_2' instead."
                    )
                return _fn
            ALL_ATTENTION_FUNCTIONS[_key] = _make_npu_fallback()
        else:
            # CUDA/CPU: re-register but return a helpful error
            def _make_import_error_fallback(n=_name):
                def _fn(*a, **kw):
                    raise RuntimeError(
                        f"Failed to load '{n}'. Ensure flash_attn and triton are installed."
                    )
                return _fn
            ALL_ATTENTION_FUNCTIONS[_key] = _make_import_error_fallback()

AttentionImplementation = Literal[tuple(ALL_ATTENTION_FUNCTIONS.keys())]


# TODO @nouamane: optimize this, and make sure it works with flashattn and flexattn
def get_attention_mask(position_ids, seq_length):
    attention_mask = torch.zeros(seq_length, seq_length, device=position_ids.device)
    start_indices = torch.where(position_ids == 0)[0]
    cu_seqlens = torch.cat(
        [start_indices, torch.tensor([seq_length], dtype=torch.int32, device=start_indices.device)]
    ).to(torch.int32)
    # make trius for each document
    for i in range(len(cu_seqlens) - 1):
        attention_mask[cu_seqlens[i] : cu_seqlens[i + 1], cu_seqlens[i] : cu_seqlens[i + 1]] = torch.tril(
            torch.ones(cu_seqlens[i + 1] - cu_seqlens[i], cu_seqlens[i + 1] - cu_seqlens[i])
        )
    return attention_mask.to(torch.bool), cu_seqlens  # [seq_length, seq_length]
