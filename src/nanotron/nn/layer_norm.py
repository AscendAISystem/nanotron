import torch
import torch.nn.functional as F
from torch import nn

try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn
except ImportError:

    def layer_norm_fn(
        x, weight, bias=None, residual=None, eps=1e-5, dropout_p=0.0,
        prenorm=False, residual_in_fp32=False, is_rms_norm=False,
        return_dropout_mask=False,
    ):
        if dropout_p > 0.0:
            raise NotImplementedError("Dropout in layer_norm fallback not supported")
        if is_rms_norm:
            input_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            x = weight * x.to(input_dtype)
        else:
            x = F.layer_norm(x, x.shape[-1:], weight, bias, eps)
        if residual is not None:
            if residual_in_fp32:
                residual = residual.to(torch.float32)
            x = (x + residual).to(x.dtype)
        if prenorm:
            return x, residual
        return x


class TritonLayerNorm(nn.LayerNorm):
    def forward(
        self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):

        return layer_norm_fn(
            input,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=False,
            return_dropout_mask=return_dropout_mask,
        )


# This is equivalent to LLaMA RMSNorm
# https://github.com/huggingface/transformers/blob/28952248b19db29ca25ccf34a5eec413376494a9/src/transformers/models/llama/modeling_llama.py#L112
class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(
        self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):

        return layer_norm_fn(
            input,
            self.weight,
            None,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True,
            return_dropout_mask=return_dropout_mask,
        )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, input):
        input_dtype = input.dtype
        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps)
        return self.weight * input.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
