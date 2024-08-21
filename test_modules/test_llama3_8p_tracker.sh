#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
device_num=8

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
task_name="logs_test_llama3_8p"
output_dir=$task_name
optim="zero2"

master_port=9001


msrun --bind_core=True --worker_num=$device_num --local_worker_num=$device_num --master_port=$master_port --log_dir=$output_dir \
python test_modules/test_llama3.py \
    --device_target Ascend \
    --is_distribute True \
    --max_device_memory 59GB \
    --enable_fa True \
    --amp_level O2 \
    --force_param_fp16 True \
    --gradient_checkpointing True \
    --optim $optim \
    --shard_size $device_num \
    \
    --enable_tracker True \
    \
    > .log_msrun.txt 2>&1 &
