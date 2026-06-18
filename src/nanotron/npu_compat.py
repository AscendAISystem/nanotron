import contextlib
import functools
import os
import warnings
from typing import Any, Callable, Iterator, Optional, Union

import torch


_HAS_NPU: Optional[bool] = None
_HAS_CUDA: Optional[bool] = None
_DEVICE_MOD: Any = None


def is_npu_available() -> bool:
    global _HAS_NPU
    if _HAS_NPU is not None:
        return _HAS_NPU
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        _HAS_NPU = False
        return _HAS_NPU
    if hasattr(torch, "npu") and hasattr(torch.npu, "is_available"):
        _HAS_NPU = torch.npu.is_available()
    elif hasattr(torch_npu, "is_npu_available"):
        _HAS_NPU = torch_npu.is_npu_available()
    else:
        _HAS_NPU = False
    return _HAS_NPU


def is_cuda_available() -> bool:
    global _HAS_CUDA
    if _HAS_CUDA is not None:
        return _HAS_CUDA
    _HAS_CUDA = torch.cuda.is_available()
    return _HAS_CUDA


def get_device_type() -> str:
    if is_npu_available():
        return "npu"
    if is_cuda_available():
        return "cuda"
    return "cpu"


def get_default_device() -> torch.device:
    return torch.device(get_device_type())


def device_mod():
    global _DEVICE_MOD
    if _DEVICE_MOD is not None:
        return _DEVICE_MOD
    if is_npu_available():
        _DEVICE_MOD = torch.npu
    elif is_cuda_available():
        _DEVICE_MOD = torch.cuda
    else:
        _DEVICE_MOD = None
    return _DEVICE_MOD


def get_backend() -> str:
    return "hccl" if is_npu_available() else "nccl"


def to_device_string() -> str:
    dev = get_device_type()
    if dev == "cpu":
        return "cpu"
    return f"{dev}:{current_device()}"


def current_device() -> int:
    mod = device_mod()
    if mod is not None:
        return mod.current_device()
    return -1


def device_count() -> int:
    mod = device_mod()
    if mod is not None:
        return mod.device_count()
    return 0


_NPU_DEVICE_MAP: Optional[dict] = None
_NPU_BROKEN_DEVICES: list = []


def _build_npu_device_map() -> dict:
    """Build a mapping of local_rank -> physical device, skipping broken devices.

    Ascend env var ASCEND_RT_VISIBLE_DEVICES is respected (native support via
    torch_npu). When set, device_count() already reflects the filtered set, but
    individual devices within that set may still be broken — we probe each one.
    Without the env var, all physical devices are probed and broken ones skipped.
    """
    global _NPU_DEVICE_MAP, _NPU_BROKEN_DEVICES
    if _NPU_DEVICE_MAP is not None:
        return _NPU_DEVICE_MAP
    mod = device_mod()
    if mod is None:
        _NPU_DEVICE_MAP = {}
        return _NPU_DEVICE_MAP

    total = mod.device_count()
    physical = []
    _NPU_BROKEN_DEVICES = []
    for i in range(total):
        try:
            mod.set_device(i)
            physical.append(i)
        except Exception as e:
            _NPU_BROKEN_DEVICES.append(i)
            warnings.warn(f"Skipping damaged NPU device {i}: {e}")
    _NPU_DEVICE_MAP = {idx: phys for idx, phys in enumerate(physical)}
    if _NPU_BROKEN_DEVICES:
        warnings.warn(
            f"Broken NPU devices detected: {_NPU_BROKEN_DEVICES}. "
            f"Mapping: {_NPU_DEVICE_MAP}. "
            f"Use ASCEND_RT_VISIBLE_DEVICES to exclude broken devices."
        )
    return _NPU_DEVICE_MAP
    return _NPU_DEVICE_MAP


def set_device(device_id: int):
    mod = device_mod()
    if mod is not None:
        dev_type = get_device_type()
        if dev_type == "npu":
            mapping = _build_npu_device_map()
            phys = mapping.get(device_id)
            if phys is None:
                raise RuntimeError(
                    f"No physical NPU device available for logical index {device_id}. "
                    f"Available mappings: {mapping}"
                )
            if phys != device_id:
                mod.set_device(phys)
                return
        mod.set_device(device_id)
    else:
        warnings.warn("No accelerator device available to set_device.")


def empty_cache():
    mod = device_mod()
    if mod is not None:
        mod.empty_cache()


def synchronize(device_id: Optional[int] = None):
    mod = device_mod()
    if mod is not None:
        mod.synchronize(device_id)


def memory_allocated(device_id: Optional[int] = None) -> int:
    mod = device_mod()
    if mod is not None:
        return mod.memory_allocated(device_id)
    return 0


def max_memory_allocated(device_id: Optional[int] = None) -> int:
    mod = device_mod()
    if mod is not None:
        return mod.max_memory_allocated(device_id)
    return 0


def max_memory_reserved(device_id: Optional[int] = None) -> int:
    mod = device_mod()
    if mod is not None:
        return mod.max_memory_reserved(device_id)
    return 0


def reset_peak_memory_stats(device_id: Optional[int] = None):
    mod = device_mod()
    if mod is not None:
        mod.reset_peak_memory_stats(device_id)


def manual_seed(seed: int):
    torch.manual_seed(seed)
    mod = device_mod()
    if mod is not None:
        mod.manual_seed(seed)
        if hasattr(mod, "manual_seed_all"):
            mod.manual_seed_all(seed)


def get_rng_state(device_id: Optional[int] = None) -> torch.Tensor:
    mod = device_mod()
    if mod is not None:
        return mod.get_rng_state(device_id) if device_id is not None else mod.get_rng_state()
    return torch.get_rng_state()


def set_rng_state(state: torch.Tensor, device_id: Optional[int] = None):
    mod = device_mod()
    if mod is not None:
        if device_id is not None:
            mod.set_rng_state(state, device_id)
        else:
            mod.set_rng_state(state)
    else:
        torch.set_rng_state(state)


@contextlib.contextmanager
def device_ctx(device_id: int) -> Iterator[None]:
    mod = device_mod()
    if mod is not None and hasattr(mod, "device"):
        with mod.device(device_id):
            yield
    else:
        yield


@contextlib.contextmanager
def disable_fp8() -> Iterator[None]:
    yield


@contextlib.contextmanager
def autocast(enabled: bool = True, dtype: Optional[torch.dtype] = None):
    if dtype is None:
        dtype = torch.bfloat16
    mod = device_mod()
    if mod is not None and hasattr(mod, "amp"):
        with mod.amp.autocast(enabled=enabled, dtype=dtype):
            yield
    else:
        yield


class Event:
    def __init__(self, enable_timing: bool = False):
        self.device_type = get_device_type()
        mod = device_mod()
        try:
            if mod is not None and hasattr(mod, "Event"):
                self._event = mod.Event(enable_timing=enable_timing)
            else:
                self._event = None
        except Exception:
            self._event = None

    def record(self, stream=None) -> "Event":
        if self._event is not None:
            self._event.record(stream)
        return self

    def wait(self, stream=None):
        if self._event is not None:
            self._event.wait(stream)

    def synchronize(self):
        if self._event is not None:
            self._event.synchronize()

    def elapsed_time(self, end_event: "Event") -> float:
        if self._event is not None and end_event._event is not None:
            return self._event.elapsed_time(end_event._event)
        return 0.0


__all__ = [
    "device_ctx",
    "is_npu_available",
    "is_cuda_available",
    "get_device_type",
    "get_default_device",
    "device_mod",
    "get_backend",
    "to_device_string",
    "current_device",
    "device_count",
    "set_device",
    "empty_cache",
    "synchronize",
    "memory_allocated",
    "max_memory_allocated",
    "max_memory_reserved",
    "reset_peak_memory_stats",
    "manual_seed",
    "get_rng_state",
    "set_rng_state",
    "disable_fp8",
    "autocast",
    "Event",
]
