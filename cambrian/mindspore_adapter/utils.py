import mindspore as ms
from mindspore import nn, context, ParallelMode


def _is_parallel():
    is_parallel = context.get_auto_parallel_context("parallel_mode") not in \
                  (ParallelMode.STAND_ALONE,)
    return is_parallel
