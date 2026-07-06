__version__ = "0.4"

# NPU 设备抽象层
from nanotron.npu_utils import (
    get_current_device,
    get_device_handle,
    get_rng_state,
    is_npu_available,
    manual_seed,
    set_rng_state,
    to_device,
)

__all__ = [
    "get_current_device",
    "get_device_handle",
    "get_rng_state",
    "is_npu_available",
    "manual_seed",
    "set_rng_state",
    "to_device",
]
