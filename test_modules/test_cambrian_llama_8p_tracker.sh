#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
device_num=8
# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
# device_num=4

export HCCL_DETERMINISTIC=true
export ASCEND_LAUNCH_BLOCKING=1
export MS_ENABLE_NUMA=0
export GLOG_v=2


# for tracker
#export MS_MEMORY_STATISTIC=1
#export MS_DEV_RUNTIME_CONF="synchronize:True"
export MS_MEMORY_STATISTIC=2
export MS_MEMORY_TRACE_PATH="./trackers"
export MS_DEV_RUNTIME_CONF="synchronize:True,memory_statistics:True,compile_statistics:True,multi_stream:False"
export MS_ALLOC_CONF="memory_tracker:True"  # enable_vmm:True


# hyper-parameters
task_name="logs_test_cambrian_llama_8p"
output_dir=$task_name
vision_tower_index="0,1,2,3"

jit_level="O2"
optim="zero2"
enable_fuse=True
enable_group=True
force_param_fp16=True
run_steps=10

master_port=9001


msrun --bind_core=True --worker_num=$device_num --local_worker_num=$device_num --master_port=$master_port --log_dir=$output_dir \
python -u test_modules/test_cambrian_llama.py \
    --device_target Ascend \
    --jit_level $jit_level \
    --is_distribute True \
    --max_device_memory 59GB \
    --enable_fa True \
    --amp_level O2 \
    --force_param_fp16 $force_param_fp16 \
    --gradient_checkpointing True \
    --optim $optim \
    --shard_size $device_num \
    --enable_fuse $enable_fuse \
    --enable_group $enable_group \
    --vision_tower_index $vision_tower_index \
    --run_steps $run_steps \
    \
    --enable_tracker True \
    \
    > .log_msrun.txt 2>&1 &
