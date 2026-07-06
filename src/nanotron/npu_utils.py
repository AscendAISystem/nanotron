# Copyright (c) 2024, HuggingFace Inc.
# SPDX-License-Identifier: Apache-2.0
#
# NPU 适配 — 设备抽象层
# 提供统一的设备管理和检测接口，支持 NPU（Ascend）/ CUDA / CPU 三种后端。
# CUDA/CPU 路径保持不动，NPU 路径通过 import guard + 运行时检测隔离。

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import torch

# ---------------------------------------------------------------------------
# NPU 可用性检测
# ---------------------------------------------------------------------------

_NPU_AVAILABLE: Optional[bool] = None


def is_npu_available() -> bool:
    """检测 torch_npu 是否可用（即当前环境是否安装了昇腾 NPU 支持库）。

    Returns:
        bool: NPU 可用返回 True，否则 False。
    """
    global _NPU_AVAILABLE
    if _NPU_AVAILABLE is not None:
        return _NPU_AVAILABLE

    try:
        import torch_npu  # noqa: F401

        _NPU_AVAILABLE = torch.npu.is_available()
    except (ImportError, RuntimeError, Exception):
        _NPU_AVAILABLE = False

    return _NPU_AVAILABLE


# ---------------------------------------------------------------------------
# 设备句柄
# ---------------------------------------------------------------------------

_DEVICE_HANDLE = None


def get_device_handle():
    """返回当前可用设备的模块句柄。

    优先级：NPU > CUDA > CPU（CPU 返回 torch.cuda 兼容桩）。

    Returns:
        module: torch.npu（NPU 可用时）、torch.cuda（CUDA 可用时）、
                或一个兼容桩对象（纯 CPU 环境）。
    """
    global _DEVICE_HANDLE
    if _DEVICE_HANDLE is not None:
        return _DEVICE_HANDLE

    if is_npu_available():
        import torch_npu  # noqa: F401

        _DEVICE_HANDLE = torch.npu
    elif torch.cuda.is_available():
        _DEVICE_HANDLE = torch.cuda
    else:
        # CPU 回退：提供一个最小兼容桩
        class _CpuDeviceHandle:
            """CPU 回退的设备句柄，提供 .is_available() / .device_count() 等常用方法。"""

            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def device_count() -> int:
                return 0

            @staticmethod
            def current_device() -> int:
                return 0

            @staticmethod
            def set_device(device: str) -> None:
                pass

            @staticmethod
            def manual_seed(seed: int) -> None:
                torch.manual_seed(seed)

            @staticmethod
            def get_rng_state() -> torch.Tensor:
                return torch.get_rng_state()

            @staticmethod
            def set_rng_state(state: torch.Tensor) -> None:
                torch.set_rng_state(state)

            @staticmethod
            def synchronize() -> None:
                pass

            @staticmethod
            def empty_cache() -> None:
                pass

            @staticmethod
            def memory_allocated() -> int:
                return 0

            @staticmethod
            def max_memory_allocated() -> int:
                return 0

            @staticmethod
            def max_memory_reserved() -> int:
                return 0

            @staticmethod
            def reset_peak_memory_stats() -> None:
                pass

            @staticmethod
            def Event(*args, **kwargs):
                return torch.cuda.Event(*args, **kwargs)

            @staticmethod
            def get_device_name(device: Optional[int] = None) -> str:
                return "cpu"

            @property
            def __name__(self) -> str:
                return "torch.cuda"

        _DEVICE_HANDLE = _CpuDeviceHandle()

    return _DEVICE_HANDLE


# ---------------------------------------------------------------------------
# 当前设备对象
# ---------------------------------------------------------------------------

def get_current_device() -> torch.device:
    """返回当前使用的设备对象。

    优先级：NPU > CUDA > CPU。

    Returns:
        torch.device: 当前设备（如 ``device(type='npu', index=0)``）。
    """
    if is_npu_available():
        import torch_npu  # noqa: F401

        device_index = torch.npu.current_device()
        return torch.device(f"npu:{device_index}")
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 张量设备移动
# ---------------------------------------------------------------------------

def to_device(
    tensor: Union[torch.Tensor, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> Union[torch.Tensor, torch.nn.Module]:
    """将张量或模块移动到指定设备。

    如果 ``device`` 为 None，则自动移动到 ``get_current_device()`` 返回的设备。

    Args:
        tensor: 要移动的张量或模块。
        device: 目标设备。为 None 时使用当前设备。
        non_blocking: 是否异步移动（仅对张量有效）。

    Returns:
        移动到目标设备后的张量或模块。
    """
    if device is None:
        device = get_current_device()
    return tensor.to(device=device, non_blocking=non_blocking)


# ---------------------------------------------------------------------------
# 随机种子管理
# ---------------------------------------------------------------------------

def manual_seed(seed: int) -> None:
    """统一设置随机种子，覆盖 CPU / CUDA / NPU。

    Args:
        seed: 随机种子值。
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if is_npu_available():
        import torch_npu  # noqa: F401

        torch.npu.manual_seed_all(seed)


def get_rng_state() -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """获取所有设备的 RNG 状态。

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            (cpu_state, cuda_state, npu_state)。
            不存在的后端对应 None。
    """
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    npu_state = None
    if is_npu_available():
        import torch_npu  # noqa: F401

        npu_state = torch.npu.get_rng_state()
    return cpu_state, cuda_state, npu_state


def set_rng_state(
    cpu_state: torch.Tensor,
    cuda_state: Optional[torch.Tensor] = None,
    npu_state: Optional[torch.Tensor] = None,
) -> None:
    """恢复所有设备的 RNG 状态。

    Args:
        cpu_state: CPU RNG 状态。
        cuda_state: CUDA RNG 状态（若 CUDA 可用）。
        npu_state: NPU RNG 状态（若 NPU 可用）。
    """
    torch.set_rng_state(cpu_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_state)
    if npu_state is not None and is_npu_available():
        import torch_npu  # noqa: F401

        torch.npu.set_rng_state(npu_state)
