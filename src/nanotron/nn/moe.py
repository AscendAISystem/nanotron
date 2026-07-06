from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import CheckpointFunction

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import ParallelismArgs
from nanotron.npu_utils import get_current_device
from nanotron.config.models_config import Qwen2Config
from nanotron.models.base import ignore_init_on_device_and_dtype
from nanotron.nn.activations import ACT2FN

logger = logging.get_logger(__name__)


def _get_grouped_gemm_ops():
    """Lazy import of grouped_gemm.ops — only resolved when MoE is actually used.
    Falls back to pure PyTorch implementation on NPU where CUDA-based grouped_gemm
    is not available."""
    try:
        import grouped_gemm.ops as _ops

        return _ops
    except ImportError:
        # Pure PyTorch fallback for NPU (grouped_gemm requires CUDA)
        return _NPUGroupedGemmOps


class _NPUGroupedGemmOps:
    """Pure PyTorch fallback for grouped_gemm operations used by MoE on NPU.
    
    Provides permute/gmm/unpermute methods with the same interface as grouped_gemm.ops.
    """

    @staticmethod
    def permute(hidden_states: torch.Tensor, routing_indices: torch.Tensor):
        """Group tokens by expert assignment.
        
        Args:
            hidden_states: [n_tokens, dim]
            routing_indices: [n_tokens, top_k] (int32 expert indices)
        
        Returns:
            dispatched_inputs: [n_tokens * top_k, dim] - tokens grouped by expert
            inverse_permute_mapping: [n_tokens * top_k] - maps output pos to flattened routing pos
        """
        n_tokens, top_k = routing_indices.shape
        device = hidden_states.device

        # Flatten routing indices: [n_tokens * top_k]
        flat_indices = routing_indices.view(-1)  # expert indices

        # Sort by expert (stable to maintain order within each expert)
        sorted_indices = torch.argsort(flat_indices, stable=True)

        # Build token index for each flattened position
        token_idx = torch.arange(n_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)
        gathered_token_idx = token_idx[sorted_indices]

        # dispatched_inputs: tokens grouped by expert in sorted order
        dispatched_inputs = hidden_states[gathered_token_idx]

        # inverse_permute_mapping: maps sorted position -> original flattened position
        inverse_permute_mapping = torch.empty(n_tokens * top_k, dtype=torch.long, device=device)
        inverse_permute_mapping[sorted_indices] = torch.arange(n_tokens * top_k, device=device)

        return dispatched_inputs, inverse_permute_mapping

    @staticmethod
    def gmm(inputs: torch.Tensor, weights: torch.Tensor, batch_sizes: torch.Tensor, trans_b: bool = False):
        """Grouped matrix multiplication using per-expert loops.
        
        Args:
            inputs: [total_tokens, in_features] (contiguous blocks per expert)
            weights: [num_experts, ...] 
            batch_sizes: [num_experts] token counts per expert (on CPU)
            trans_b: whether to transpose second dim of weights
        
        Returns:
            output: [total_tokens, out_features]
        """
        num_experts = weights.shape[0]
        outputs = []
        offset = 0
        for expert_id in range(num_experts):
            n = int(batch_sizes[expert_id].item())
            if n == 0:
                continue
            expert_input = inputs[offset:offset + n]
            expert_weight = weights[expert_id]
            if trans_b:
                expert_output = torch.matmul(expert_input, expert_weight.t())
            else:
                expert_output = torch.matmul(expert_input, expert_weight)
            outputs.append(expert_output)
            offset += n

        if len(outputs) == 0:
            return inputs.new_zeros(0, weights.shape[-1])
        return torch.cat(outputs, dim=0)

    @staticmethod
    def unpermute(outputs: torch.Tensor, inverse_permute_mapping: torch.Tensor, routing_weights: torch.Tensor):
        """Reverse the permute operation: scatter weighted outputs back to original positions.
        
        Args:
            outputs: [total_expert_tokens, dim] - expert outputs grouped by expert
            inverse_permute_mapping: [total_expert_tokens] - maps output pos -> flattened routing pos
            routing_weights: [n_tokens, top_k] - routing weights
        
        Returns:
            combined: [n_tokens, dim]
        """
        n_tokens, top_k = routing_weights.shape
        dim = outputs.shape[-1]
        device = outputs.device

        # Unsort back to original flattened order: [n_tokens * top_k, dim]
        flat_outputs = torch.zeros(n_tokens * top_k, dim, dtype=outputs.dtype, device=device)
        flat_outputs[inverse_permute_mapping] = outputs

        # Apply routing weights and reshape
        flat_outputs = flat_outputs.view(n_tokens, top_k, dim)
        weights = routing_weights.unsqueeze(-1)  # [n_tokens, top_k, 1]

        # Weighted sum over top_k for each token
        combined = (flat_outputs * weights).sum(dim=1)

        return combined


ops = None  # placeholder, resolved lazily via _get_grouped_gemm_ops()


class Router(nn.Module):
    def __init__(
        self, config: Qwen2Config, parallel_config: Optional[ParallelismArgs], tp_pg: dist.ProcessGroup, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.parallel_config = parallel_config
        self.tp_pg = tp_pg
        self.layer_idx = layer_idx

        self.num_experts = config.moe_config.num_experts
        self.num_experts_per_token = config.moe_config.top_k

        # float32 routing weights
        # NOTE: qwen keep the routing weights in float32
        # https://github.com/huggingface/transformers/blob/27a25bee4fcb865e8799ba026f1ea4455f2cca98/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L608
        with ignore_init_on_device_and_dtype():
            self.weight = nn.Parameter(
                torch.randn(self.num_experts, config.hidden_size, dtype=torch.float32, device=get_current_device())
            )
        assert self.weight.dtype == torch.float32

    def gating(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for all experts (no softmax)."""
        # NOTE: qwen keep the routing logits in float32
        # https://github.com/huggingface/transformers/blob/27a25bee4fcb865e8799ba026f1ea4455f2cca98/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L613
        return F.linear(x.to(torch.float32), self.weight, bias=None)

    def routing(self, logits: torch.Tensor):
        """Top-k softmax-normalized routing weights and indices."""
        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        routing_weights, routing_indices = torch.topk(routing_weights, k=self.num_experts_per_token, dim=-1)
        routing_indices = routing_indices.to(torch.int32)  # NOTE: ops.permute requires indices to be int32
        return routing_weights, routing_indices

    def forward(self, x: torch.Tensor):
        logits = self.gating(x)
        return self.routing(logits)


class GroupedMLP(nn.Module):
    def __init__(self, config: Qwen2Config, parallel_config: Optional[ParallelismArgs]):
        super().__init__()

        num_local_experts = config.moe_config.num_experts // parallel_config.expert_parallel_size
        self.merged_gate_up_proj = nn.Parameter(
            torch.randn(num_local_experts, config.hidden_size, 2 * config.moe_config.moe_intermediate_size)
        )
        self.merged_down_proj = nn.Parameter(
            torch.randn(num_local_experts, config.moe_config.moe_intermediate_size, config.hidden_size)
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ):
        """
        assume hidden_states is permuted

        grouped_gemm's notes:
        ops.gemm expect the inputs to have the following criteria:
        + expect a, b are in bfloat16
        + expect num_tokens_per_expert is a on cpu
        """
        # NOTE: ops.gemm requires "batch_sizes" (aka: num_tokens_per_expert here) to be on cpu
        num_tokens_per_expert = num_tokens_per_expert.to("cpu")
        merged_states = _get_grouped_gemm_ops().gmm(hidden_states, self.merged_gate_up_proj, num_tokens_per_expert, trans_b=False)
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        hidden_states = self.act(gate_states) * up_states
        hidden_states = _get_grouped_gemm_ops().gmm(hidden_states, self.merged_down_proj, num_tokens_per_expert, trans_b=False)

        return {"hidden_states": hidden_states}


class Qwen2MoELayer(nn.Module):
    """Mixture of experts Layer for Qwen2 models."""

    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # MoE specific configurations
        self.num_experts = config.moe_config.num_experts  # Total number of experts
        self.num_local_experts = (
            config.moe_config.num_experts // parallel_config.expert_parallel_size
        )  # Experts per device
        self.num_experts_per_token = config.moe_config.top_k  # Number of experts used per token (top-k)
        self.expert_parallel_size = parallel_config.expert_parallel_size
        self.num_local_experts = self.num_experts // self.expert_parallel_size  # Experts per device

        # Get TP mode configuration

        # Router for selecting experts
        self.router = Router(config, parallel_config, tp_pg, layer_idx)

        # Enable shared experts if configured
        self.enable_shared_expert = config.moe_config.enable_shared_expert
        if self.enable_shared_expert:
            from nanotron.models.qwen import Qwen2MLP

            self.shared_expert = Qwen2MLP(
                config=config,
                parallel_config=parallel_config,
                tp_pg=tp_pg,
                intermediate_size=config.moe_config.shared_expert_intermediate_size,
            )
            # TODO: duplicte the shared expert gate
            self.shared_expert_gate = nn.Linear(
                self.hidden_size,
                1,
                bias=False,
            )  # TODO: ensure shared_expert_gate is tied across TP

        # Create the expert MLPs
        self.experts = GroupedMLP(config, parallel_config)
        # Whether to recompute MoE layer during backward pass for memory efficiency
        self.recompute_layer = parallel_config.recompute_layer

    def _dispatch_tokens(
        self,
        hidden_states: torch.Tensor,
        routing_indices: torch.Tensor,
    ):
        """
        Dispatches tokens to their selected experts.
        In a full implementation, this would handle the actual token routing logic
        including communication between devices.
        """
        # NOTE: start from expert 0 to expert n
        num_tokens_per_expert = torch.bincount(
            routing_indices.flatten(), minlength=self.num_local_experts
        )  # [num_local_experts]
        dispatched_inputs, inverse_permute_mapping = _get_grouped_gemm_ops().permute(hidden_states, routing_indices)
        return dispatched_inputs, inverse_permute_mapping, num_tokens_per_expert

    def _combine_expert_outputs(self, expert_outputs, inverse_mapping, routing_weights):
        """
        Combines outputs from different experts back to the original tensor layout.
        """
        hidden_states = _get_grouped_gemm_ops().unpermute(expert_outputs, inverse_mapping, routing_weights)
        return hidden_states

    def _core_forward(self, hidden_states):
        """Core forward logic for MoE layer."""
        # Get top-k routing weights and indices
        routing_weights, routing_indices = self.router(hidden_states)  # [num_tokens, num_experts_per_token]

        # Dispatch tokens to experts
        dispatched_inputs, inverse_permute_mapping, num_tokens_per_expert = self._dispatch_tokens(
            hidden_states, routing_indices
        )

        expert_outputs = self.experts(dispatched_inputs, num_tokens_per_expert)

        output = self._combine_expert_outputs(
            expert_outputs["hidden_states"], inverse_permute_mapping, routing_weights
        )

        # Add shared expert contribution if enabled
        if self.enable_shared_expert:
            shared_expert_output = self.shared_expert(hidden_states=hidden_states)["hidden_states"]
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            output = output + shared_gate * shared_expert_output

        return output

    def _checkpointed_forward(self, hidden_states):
        """Apply gradient checkpointing to save memory during training."""
        return CheckpointFunction.apply(self._core_forward, True, hidden_states)

    def forward(self, hidden_states):
        """Forward pass for the MoE layer."""
        if self.recompute_layer and self.training:
            hidden_states = self._checkpointed_forward(hidden_states)
        else:
            hidden_states = self._core_forward(hidden_states)

        return {"hidden_states": hidden_states}
