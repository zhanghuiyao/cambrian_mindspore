import mindspore as ms
from mindspore import nn, context, ParallelMode


_DTYPE_2_STRING = {
    ms.float16: "float16",
    ms.bfloat16: "bfloat16",
    ms.float32: "float32",
    ms.float64: "float64",
    ms.uint8: "uint8",
    ms.int8: "int8",
    ms.int16: "int16",
    ms.int32: "int32",
    ms.int64: "int64",
    ms.bool_:  "bool",
}


def _is_parallel():
    is_parallel = context.get_auto_parallel_context("parallel_mode") not in \
                  (ParallelMode.STAND_ALONE,)
    return is_parallel
