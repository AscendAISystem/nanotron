import warnings

from nanotron.npu_compat import is_npu_available

from nanotron.fp8.dtypes import DTypes  # noqa
from nanotron.fp8.linear import FP8Linear  # noqa
from nanotron.fp8.parameter import FP8Parameter  # noqa
from nanotron.fp8.tensor import FP8Tensor  # noqa

if not is_npu_available():
    try:
        import transformer_engine  # noqa
        import transformer_engine_extensions  # noqa
    except ImportError:
        warnings.warn("Please install Transformer engine for FP8 training!")
