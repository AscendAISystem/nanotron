import torch
from torch import nn
from einops import rearrange
from nanotron import logging
from nanotron.logging import warn_once
from nanotron.npu_utils import get_current_device, is_npu_available

logger = logging.get_logger(__name__)

# flash_attn is CUDA-only; guard import for NPU environments
try:
    from flash_attn.layers.rotary import apply_rotary_emb as flash_apply_rotary_emb
    from flash_attn.layers.rotary import RotaryEmbedding as OrigFlashRotaryEmbedding

    _flash_attn_rotary_available = True
except ImportError:
    flash_apply_rotary_emb = None
    OrigFlashRotaryEmbedding = None
    _flash_attn_rotary_available = False


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: float = 10000.0,
        interleaved: bool = False,
        seq_len_scaling_factor: float = None,
        fused: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len  # we set it as max_position_embeddings in init. but we ignore it of we provide `seq_length` in forward
        self.interleaved = interleaved
        self.seq_len_scaling_factor = seq_len_scaling_factor
        self.fused = fused
        # Generate inverse frequency buffer directly in the constructor
        self.register_buffer(
            "freqs_cis",
            1.0 / (base ** (torch.arange(0, dim, 2, device=get_current_device(), dtype=torch.float) / dim)),
            persistent=False,
        )
        # These are caches that are recomputed during inference
        self.register_buffer("cos_values", None, persistent=False)
        self.register_buffer("sin_values", None, persistent=False)

        assert self.freqs_cis.device.type == get_current_device().type

    def forward(self, seq_length=None, position_offset=0, position_ids=None):
        """Generate rotary position embeddings.

        Args:
            seq_length (int, optional): Sequence length to use. Defaults to max_seq_len.
            position_offset (int, optional): Offset for position ids. Defaults to 0.
            position_ids (Tensor, optional): Position ids to use. Defaults to None. [batch_size, seq_length]

        Returns:
            Tensor: Rotary embeddings of shape [seq_length, 1, 1, dim]
        """
        self.freqs_cis = self.freqs_cis.to(torch.float)  # TODO @nouamane: Fix using `DTypeInvariantTensor` ...

        # Generate position indices
        if position_ids is not None:
            assert seq_length is None, "seq_length must be None if position_ids is provided"
            assert position_offset == 0, "position_offset must be 0 if position_ids is provided"
            # TODO @nouamane: Using position_ids means we compute redundant embeddings for same positions
            positions = position_ids.to(device=self.freqs_cis.device, dtype=self.freqs_cis.dtype)  # [b*s]
            self.max_seq_len = positions.max() + 1
        else:
            seq_length = seq_length or self.max_seq_len
            positions = (
                torch.arange(seq_length, device=self.freqs_cis.device, dtype=self.freqs_cis.dtype) + position_offset
            )  # [seq_length]
            self.max_seq_len = seq_length

        # Apply sequence length scaling if specified
        if self.seq_len_scaling_factor is not None:
            positions = positions / self.seq_len_scaling_factor

        # Compute position frequencies
        # TODO @nouamane: Using position_ids means we compute redundant embeddings for same positions. Only use them in SFT
        position_freqs = torch.outer(positions, self.freqs_cis)  # [seq_length, dim/2]

        # Organize embeddings based on interleaving strategy
        if self.fused:
            embeddings = position_freqs  # [b*s, dim/2] or [seq_length, dim/2]
        else:
            if not self.interleaved:
                embeddings = torch.cat((position_freqs, position_freqs), dim=-1)  # [b*s, dim] or [seq_length, dim]
            else:
                embeddings = torch.stack(
                    (position_freqs.view(-1, 1), position_freqs.view(-1, 1)), dim=-1
                )  # [b*s*dim, 2] or [seq_length*dim, 2]
                embeddings = embeddings.view(position_freqs.shape[0], -1)  # [b*s, dim] or [seq_length, dim]

        return embeddings  # [b*s, dim] or [seq_length, dim] or [b*s, dim/2] or [seq_length, dim/2]

    def rotate_half(self, x):
        """Rotates half the hidden dimensions of the input tensor."""
        if self.interleaved:
            even_dims = x[..., ::2]
            odd_dims = x[..., 1::2]
            return torch.cat((-odd_dims, even_dims), dim=-1)
        else:
            first_half = x[..., : x.shape[-1] // 2]
            second_half = x[..., x.shape[-1] // 2 :]
            return torch.cat((-second_half, first_half), dim=-1)

    def apply_rotary_pos_emb(self, tensor, freqs, multi_latent_attention=False, mscale=1.0, seq_length=None):
        """Apply rotary positional embedding to input tensor.

        Args:
            tensor (Tensor): Input tensor of shape [..., dim] if not fused, [batch_size*seq_length, nheads, dim] if fused
            freqs (Tensor, optional): Pre-computed position embeddings [..., dim] same or broadcastable to tensor
            multi_latent_attention (bool): Whether to use multi-latent attention
            mscale (float): Scaling factor for rotary embeddings

        Returns:
            Tensor: The input tensor after applying rotary positional embedding
        """
        rotary_dim = freqs.shape[-1]

        # Split the tensor for rotary embedding application
        if freqs.shape[-1] != rotary_dim:
            rotary_part, pass_through_part = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
        else:
            rotary_part, pass_through_part = tensor, None

        # Handle multi-latent attention
        if multi_latent_attention:
            x1 = rotary_part[..., 0::2]
            x2 = rotary_part[..., 1::2]
            rotary_part = torch.cat((x1, x2), dim=-1)

        # Get cosine and sine components with scaling
        if self.cos_values is None:
            self.cos_values = (torch.cos(freqs) * mscale).to(tensor.dtype)
            self.sin_values = (torch.sin(freqs) * mscale).to(tensor.dtype)

        # Apply rotary embedding
        rotary_part = rotary_part.view(
            -1, seq_length, rotary_part.shape[1], rotary_part.shape[2]
        )  # [b, s, nheads, dim/2]
        if self.fused:
            if is_npu_available():
                # NPU: flash_attn unavailable, use pure PyTorch rotation
                rotated_tensor = (rotary_part * self.cos_values.unsqueeze(1)) + (
                    self.rotate_half(rotary_part) * self.sin_values.unsqueeze(1)
                )
            else:
                rotated_tensor = flash_apply_rotary_emb(
                    rotary_part, self.cos_values, self.sin_values, interleaved=self.interleaved, inplace=True
                )
            # TODO @nouamane: support cu_seqlens from position_ids
        else:
            rotated_tensor = (rotary_part * self.cos_values.unsqueeze(1)) + (
                self.rotate_half(rotary_part) * self.sin_values.unsqueeze(1)
            )

        # Concatenate with the pass-through part (if any)
        if pass_through_part is not None and pass_through_part.shape[-1] > 0:
            return torch.cat((rotated_tensor, pass_through_part), dim=-1)
        return rotated_tensor
    
if _flash_attn_rotary_available:

    class FlashRotaryEmbedding(OrigFlashRotaryEmbedding):

        def __init__(
            self,
            dim: int,
            base=10000.0,
            interleaved=False,
            scale_base=None,
            pos_idx_in_fp32=True,
            device=None,
            seq_len_interpolation_factor=None,
        ):
            super().__init__(
                dim,
                base,
                interleaved,
                scale_base,
                pos_idx_in_fp32,
                device,
            )
            self.seq_len_interpolation_factor = seq_len_interpolation_factor

        def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
            # Reset the tables if the sequence length has changed,
            # if we're on a new device (possibly due to tracing for instance),
            # or if we're switching from inference mode to training
            if (
                seqlen > self._seq_len_cached
                or self._cos_cached is None
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
                or (self.training and self._cos_cached.is_inference())
            ):
                self._seq_len_cached = seqlen
                # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
                # And the output of arange can be quite large, so bf16 would lose a lot of precision.
                # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
                if self.pos_idx_in_fp32:
                    t = torch.arange(seqlen, device=device, dtype=torch.float32)
                    # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                    # will be large. Having it in bf16 will lose a lot of precision and cause the
                    # cos & sin output to change significantly.
                    # We want to recompute self.inv_freq if it was not loaded in fp32
                    if self.inv_freq.dtype != torch.float32:
                        inv_freq = self._compute_inv_freq(device=device)
                    else:
                        inv_freq = self.inv_freq
                else:
                    t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                    inv_freq = self.inv_freq

                # fixed linear scaling
                if self.seq_len_interpolation_factor is not None:
                    warn_once(f"seq_len_interpolation_factor is set to {self.seq_len_interpolation_factor}", logger, rank=0)
                    t *= 1 / self.seq_len_interpolation_factor

                # Don't do einsum, it converts fp32 to fp16 under AMP
                # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
                freqs = torch.outer(t, inv_freq)
                if self.scale is None:
                    self._cos_cached = torch.cos(freqs).to(dtype)
                    self._sin_cached = torch.sin(freqs).to(dtype)
                else:
                    power = (
                        torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                        - seqlen // 2
                    ) / self.scale_base
                    scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                    # We want the multiplication by scale to happen in fp32
                    self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                    self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                    self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                    self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

else:

    class FlashRotaryEmbedding(RotaryEmbedding):
        """NPU fallback: flash_attn not available, use pure PyTorch RotaryEmbedding.

        Maps FlashRotaryEmbedding parameters to RotaryEmbedding for NPU compatibility.
        The _update_cos_sin_cache method is a no-op since RotaryEmbedding computes
        cos/sin on-the-fly in forward().
        """

        def __init__(
            self,
            dim: int,
            base=10000.0,
            interleaved=False,
            scale_base=None,
            pos_idx_in_fp32=True,
            device=None,
            seq_len_interpolation_factor=None,
        ):
            super().__init__(
                dim=dim,
                max_seq_len=0,
                base=base,
                interleaved=interleaved,
                seq_len_scaling_factor=seq_len_interpolation_factor,
                fused=False,
            )

        def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
            """No-op: RotaryEmbedding(base class) computes cos/sin in forward()."""
            pass

        def forward(self, x, *args, **kwargs):
            """Flash-attn compatible forward: apply rotary embedding.

            Handles both calling conventions:
              - rotary_emb(x, seqlen_offset=0, max_seqlen=None)   [single tensor]
              - rotary_emb(q, kv, seqlen_offset=0, max_seqlen=...) [packed q,kv]

            Args:
                x: Query tensor or first positional arg.
                *args: If provided, args[0] is kv tensor (packed call).
                seqlen_offset: Offset for position ids (default 0).
                max_seqlen: Maximum sequence length (unused, kept for API compat).

            Returns:
                Tensor(s) with rotary embeddings applied.
            """
            seqlen_offset = kwargs.get('seqlen_offset', 0)
            max_seqlen = kwargs.get('max_seqlen', None)

            if args:
                # Called as rotary_emb(q, kv, seqlen_offset=..., max_seqlen=...)
                q, kv = x, args[0]
                seq_length = q.shape[1]
                # Compute rotary embeddings
                freqs = super().forward(seq_length=seq_length, position_offset=seqlen_offset)
                # Apply to q
                q = self.apply_rotary_pos_emb(q, freqs, seq_length=seq_length)
                # Apply to kv: kv shape [batch, seq, 2, n_kv_heads, head_dim]
                k = kv[..., 0, :, :]  # [batch, seq, n_kv_heads, head_dim]
                v = kv[..., 1, :, :]  # [batch, seq, n_kv_heads, head_dim]
                k = self.apply_rotary_pos_emb(k, freqs, seq_length=seq_length)
                # Re-pack kv
                kv = torch.stack([k, v], dim=2)
                return q, kv
            else:
                # Single tensor (inference path)
                seq_length = x.shape[1] if x.dim() >= 3 else x.shape[0]
                freqs = super().forward(seq_length=seq_length, position_offset=seqlen_offset)
                return self.apply_rotary_pos_emb(x, freqs, seq_length=seq_length)