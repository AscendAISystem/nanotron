import datetime
import os
from functools import cache, lru_cache
from typing import Dict, List, Optional, Tuple

import torch
from packaging import version
from torch import distributed as dist
from torch.distributed import *  # noqa
from torch.distributed.distributed_c10d import ProcessGroup

from nanotron.npu_utils import is_npu_available
from nanotron.utils import find_free_port

torch_version_above_1_13 = version.parse(torch.__version__) >= version.parse("1.13.0")
Work = dist.Work if torch_version_above_1_13 else dist._Work

# Note: When debugging communication hangs, try decreasing this timeout.
default_pg_timeout = datetime.timedelta(minutes=20)


def new_group(  # pylint: disable=function-redefined
    ranks=None, timeout=default_pg_timeout, backend=None, pg_options=None
) -> ProcessGroup:
    if len(ranks) == 0:
        raise ValueError("Cannot create a group with not ranks inside it")

    return dist.new_group(ranks=ranks, timeout=timeout, backend=backend, pg_options=pg_options)


def reduce_scatter_tensor(  # pylint: disable=function-redefined
    output: torch.Tensor,
    input: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[Work]:
    if group is None:
        group = dist.torch_dist.distributed_c10d._get_default_group()

    assert (
        group.size() > 1
    ), "You should probably not call `reduce_scatter_tensor` with a single rank, as it copies data over"

    if torch_version_above_1_13:
        return dist.reduce_scatter_tensor(output=output, input=input, group=group, op=op, async_op=async_op)
    else:
        # Support pytorch 1.12
        return dist._reduce_scatter_base(output=output, input=input, group=group, op=op, async_op=async_op)


def all_gather_into_tensor(  # pylint: disable=function-redefined
    output_tensor, input_tensor, group: Optional[ProcessGroup] = None, async_op: bool = False
) -> Optional[Work]:
    if group is None:
        group = dist.torch_dist.distributed_c10d._get_default_group()

    assert (
        group.size() > 1
    ), "You should probably not call `all_gather_into_tensor` with a single rank, as it copies data over"

    if torch_version_above_1_13:
        return dist.all_gather_into_tensor(
            output_tensor=output_tensor, input_tensor=input_tensor, group=group, async_op=async_op
        )
    else:
        # Support Pytorch 1.12
        return dist.distributed_c10d._all_gather_base(
            output_tensor=output_tensor, input_tensor=input_tensor, group=group, async_op=async_op
        )


def reduce_scatter_coalesced(
    output_tensor_list: List[torch.Tensor],
    input_tensor_lists: List[List[torch.Tensor]],
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[torch._C.Future]:
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Args:
        output_tensor_list (list[Tensor]): Output tensor.
        input_tensor_lists (list[list[Tensor]]): List of tensors to reduce and scatter.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    assert len(output_tensor_list) > 0
    assert len(input_tensor_lists) == len(output_tensor_list)
    device = output_tensor_list[0].device
    dtype = output_tensor_list[0].dtype
    group_size = len(input_tensor_lists[0])

    assert (
        group_size > 1
    ), "You should probably not call `reduce_scatter_coalesced` with a single rank, as it copies data over"

    for output_tensor in output_tensor_list:
        assert device == output_tensor.device
        assert dtype == output_tensor.dtype

    for input_tensor_list in input_tensor_lists:
        assert len(input_tensor_list) == group_size, f"Expected {len(input_tensor_list)} == {group_size}"
        for input_tensor in input_tensor_list:
            assert device == input_tensor.device
            assert dtype == input_tensor.dtype

    output_tensor_buffer = torch._utils._flatten_dense_tensors(output_tensor_list)
    input_tensor_buffer_list = [
        torch._utils._flatten_dense_tensors(
            [input_tensor_list[group_rank] for input_tensor_list in input_tensor_lists]
        )
        for group_rank in range(group_size)
    ]

    work = dist.reduce_scatter(output_tensor_buffer, input_tensor_buffer_list, op=op, group=group, async_op=async_op)

    def update_output():
        for original_buffer, reduced_buffer in zip(
            output_tensor_list, torch._utils._unflatten_dense_tensors(output_tensor_buffer, output_tensor_list)
        ):
            original_buffer.copy_(reduced_buffer)

    if async_op is True:
        return work.get_future().then(lambda fut: update_output())
    else:
        # No need to run `work.wait()` since `dist.reduce_scatter` already waits
        update_output()


def all_reduce_coalesced(  # pylint: disable=function-redefined
    tensors: List[torch.Tensor],
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[torch._C.Future]:
    if group is None:
        group = dist.torch_dist.distributed_c10d._get_default_group()

    if group.size() == 1:
        return

    return dist.all_reduce_coalesced(tensors, op=op, group=group, async_op=async_op)


def all_gather_coalesced(  # pylint: disable=function-redefined
    output_tensor_lists: List[List[torch.Tensor]],
    input_tensor_list: List[torch.Tensor],
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[torch._C.Future]:
    """
    `torch` has a deprecated version of this method that doesn't work over NCCL.
    All gathers a list of tensors to all processes in a group.

    Supports mixed-dtype tensors by grouping them by dtype internally.

    Args:
        output_tensor_lists (list[list[Tensor]]): Output tensor.
        input_tensor_list (list[Tensor]): List of tensors to all_gather from.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    assert len(output_tensor_lists) > 0
    assert len(input_tensor_list) == len(output_tensor_lists)
    device = input_tensor_list[0].device
    group_size = len(output_tensor_lists[0])

    assert (
        group_size > 1
    ), "You should probably not call `all_gather_coalesced` with a single rank, as it copies data over"

    # Group by dtype to handle mixed-dtype tensors (e.g., bfloat16 + float32 params)
    dtype_groups: Dict[torch.dtype, Dict] = {}
    for idx, input_tensor in enumerate(input_tensor_list):
        dt = input_tensor.dtype
        if dt not in dtype_groups:
            dtype_groups[dt] = {"input_idxs": [], "device": input_tensor.device}
        assert device == input_tensor.device, f"Device mismatch: {device} vs {input_tensor.device}"
        dtype_groups[dt]["input_idxs"].append(idx)

    # If all same dtype, use the original fast path
    if len(dtype_groups) == 1:
        dt = input_tensor_list[0].dtype
        for output_tensor_list in output_tensor_lists:
            assert len(output_tensor_list) == group_size
            for output_tensor in output_tensor_list:
                assert device == output_tensor.device
                assert dt == output_tensor.dtype

        # Invert from `[param_idx][group_rank]` to `[group_rank][param_idx]`
        transposed = [
            [output_tensor_list[group_rank] for output_tensor_list in output_tensor_lists]
            for group_rank in range(group_size)
        ]

        input_tensor_buffer = torch._utils._flatten_dense_tensors(input_tensor_list)
        output_tensor_buffer_list = [
            torch._utils._flatten_dense_tensors(output_tensor_list) for output_tensor_list in transposed
        ]

        work = dist.all_gather(output_tensor_buffer_list, input_tensor_buffer, group=group, async_op=async_op)

        def update_output():
            for original_buffer_list, gathered_buffer_tensor in zip(transposed, output_tensor_buffer_list):
                for original_buffer, gathered_buffer in zip(
                    original_buffer_list,
                    torch._utils._unflatten_dense_tensors(gathered_buffer_tensor, original_buffer_list),
                ):
                    original_buffer.copy_(gathered_buffer)

        if async_op is True:
            return work.get_future().then(lambda fut: update_output())
        else:
            update_output()
            return None

    # Mixed dtypes: process each dtype group separately
    futures = []
    for dt, group_info in dtype_groups.items():
        idxs = group_info["input_idxs"]
        group_input = [input_tensor_list[i] for i in idxs]
        group_output = [output_tensor_lists[i] for i in idxs]

        for output_tensor_list in group_output:
            assert len(output_tensor_list) == group_size
            for output_tensor in output_tensor_list:
                assert device == output_tensor.device
                assert dt == output_tensor.dtype

        # Invert
        transposed = [
            [output_tensor_list[group_rank] for output_tensor_list in group_output]
            for group_rank in range(group_size)
        ]

        input_buffer = torch._utils._flatten_dense_tensors(group_input)
        output_buffer_list = [
            torch._utils._flatten_dense_tensors(tlist) for tlist in transposed
        ]

        work = dist.all_gather(output_buffer_list, input_buffer, group=group, async_op=async_op)

        def _make_updater(transposed_tlists, output_buf_list):
            def update_output():
                for orig_list, gathered_buf in zip(transposed_tlists, output_buf_list):
                    for orig, gathered in zip(
                        orig_list,
                        torch._utils._unflatten_dense_tensors(gathered_buf, orig_list),
                    ):
                        orig.copy_(gathered)
            return update_output

        if async_op:
            futures.append(work.get_future().then(lambda fut: _make_updater(transposed, output_buffer_list)()))
        else:
            _make_updater(transposed, output_buffer_list)()

    if async_op:
        # Return the last future (primary will still wait for all)
        return futures[-1] if futures else None
    return None


# This cache has a speedup of 4 tflops on a 7b model
@cache
def get_global_rank(group: ProcessGroup, group_rank: int) -> int:  # pylint: disable=function-redefined
    if torch_version_above_1_13:
        return dist.get_global_rank(group, group_rank=group_rank)
    else:
        # Support pytorch 1.12
        return dist.distributed_c10d._get_global_rank(group=group, rank=group_rank)


def get_global_ranks(group: ProcessGroup) -> Tuple[int]:
    return tuple(sorted((get_global_rank(group, i) for i in range(group.size()))))


# We cache for dp, pp, tp process groups, world group, and tied process group for tied params
@lru_cache
def get_rank(group: Optional[ProcessGroup] = None) -> int:  # pylint: disable=function-redefined
    """Similar to `get_rank` except we raise an exception instead of return -1 when current rank is not part of the group"""
    result = dist.get_rank(group)
    if result == -1:
        raise RuntimeError("Can not call `get_rank` on a group in which current process is not a part of")
    return result


def initialize_torch_distributed():
    """Initializes torch distributed with the environment variables"""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if is_npu_available():
        # Set the device id for NPU (Ascend).
        # `torch.npu.device_count` should return the number of device on a single node.
        # We assume the nodes to be homogeneous (same number of gpus per node)
        device_id = local_rank
        import torch_npu  # noqa: F401

        torch.npu.set_device(device_id)
        backend = "hccl"
    elif torch.cuda.is_available():
        # Set the device id.
        # `torch.cuda.device_count` should return the number of device on a single node.
        # We assume the nodes to be homogeneous (same number of gpus per node)
        device_id = local_rank
        torch.cuda.set_device(torch.cuda.device(device_id))
        backend = "nccl"
    else:
        # Fallback to gloo for CPU-only distributed training
        backend = "gloo"

    # Call the init process.

    port = os.getenv("MASTER_PORT")
    if port is None:
        port = find_free_port()
    else:
        port = int(port)

    init_method = f"env://localhost:{port}"
    dist.init_process_group(
        init_method=init_method, backend=backend, world_size=world_size, rank=rank, timeout=dist.default_pg_timeout
    )
    return True
