from dataclasses import dataclass, field
from typing import Optional

import mindspore as ms
from mindspore.communication.management import init, get_rank, get_group_size


@dataclass
class MindSporeArguments:
    # for mindspore

    mode: str = field(
        default=ms.GRAPH_MODE,
        metadata = {"help": "Graph/Pynative"}
    )

    jit_level: Optional[str] = field(
        default="O0",
        metadata={
            "help": ("jit level")
        }
    )

    device_target: str = field(
        default="Ascend",
        metadata = {"help": "Ascend/GPU/CPU"}
    )

    is_distribute: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use PyTorch Fully Sharded Data Parallel (FSDP) training (in distributed training"
                " only). The base option should be `full_shard`, `shard_grad_op` or `no_shard` and you can add"
                " CPU-offload to `full_shard` or `shard_grad_op` like this: full_shard offload` or `shard_grad_op"
                " offload`. You can add auto-wrap to `full_shard` or `shard_grad_op` with the same syntax: full_shard"
                " auto_wrap` or `shard_grad_op auto_wrap`."
            ),
        },
    )
    mix_precision: Optional[str] = field(
        default="O2",
        metadata={
            "help": (
                ""
            )
        },
    )
    zero: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Enable zero parallelism, select from [stage1, stage2, stage3]"
            )
        },
    )
    max_device_memory: Optional[str] = field(
        default=None,
        metadata={
            "help": ("max device memory")
        },
    )

    precision_mode: Optional[str] = field(
        default="must_keep_origin_dtype",
        metadata={
            "help": ("global precision_mode")
        }
    )


def init_environment(training_args: MindSporeArguments):
    # set mindspore context
    ms.set_context(
        mode=training_args.mode,
        device_target=training_args.device_target,
        jit_config={"jit_level": training_args.jit_level}
    )

    if training_args.mode == ms.PYNATIVE_MODE:
        ms.set_context(pynative_synchronize=True)
        print("WARNING: run pynative mode, set `pynative_synchronize` True")

    if training_args.max_device_memory is not None:
        ms.set_context(max_device_memory=training_args.max_device_memory)

    if training_args.precision_mode is not None:
        ms.set_context(ascend_config={"precision_mode": training_args.precision_mode},)

    if training_args.is_distribute:
        init()
        world_size = get_group_size()
        rank_id = get_rank()
        print(f"init_environment, rank_id: {rank_id}, world_size: {world_size}")

        ms.reset_auto_parallel_context()

        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=world_size,
        )
