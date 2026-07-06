import torch
try:
    import transformer_engine as te  # noqa: F401
except ImportError:
    te = None

from nanotron.fp8.constants import FP8_GPU_NAMES


def is_fp8_available() -> bool:
    """Check if FP8 is available on the current device."""
    if te is None:
        return False
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device()).lower()
        return any(gpu_name in device_name for gpu_name in FP8_GPU_NAMES)
    else:
        return False
