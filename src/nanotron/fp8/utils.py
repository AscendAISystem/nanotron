import torch

from nanotron.fp8.constants import FP8_GPU_NAMES
from nanotron.npu_compat import device_mod, is_cuda_available, is_npu_available


def is_fp8_available() -> bool:
    """Check if FP8 is available on the current device."""
    if is_npu_available():
        return False
    import transformer_engine as te  # noqa
    if is_cuda_available():
        device_name = device_mod().get_device_name(device_mod().current_device()).lower()
        return any(gpu_name in device_name for gpu_name in FP8_GPU_NAMES)
    else:
        return False
