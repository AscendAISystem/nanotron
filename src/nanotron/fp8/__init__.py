import warnings

from nanotron.fp8.dtypes import DTypes  # noqa

try:
    import transformer_engine as te  # noqa
    import transformer_engine_extensions as tex  # noqa
    _TE_AVAILABLE = True
except ImportError:
    _TE_AVAILABLE = False
    warnings.warn("Please install Transformer engine for FP8 training!")

if _TE_AVAILABLE:
    from nanotron.fp8.linear import FP8Linear  # noqa
    from nanotron.fp8.parameter import FP8Parameter  # noqa
    from nanotron.fp8.tensor import FP8Tensor  # noqa
else:
    FP8Linear = None
    FP8Parameter = None
    FP8Tensor = None
