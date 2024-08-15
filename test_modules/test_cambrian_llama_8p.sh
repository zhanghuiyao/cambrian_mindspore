#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
device_num=8
# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
# device_num=4

export MS_ENABLE_NUMA=0
export GLOG_v=2

export MS_MEMORY_STATISTIC=1
export MS_DEV_RUNTIME_CONF="synchronize:True"


# hyper-parameters
task_name="logs_test_cambrian_llama_8p"
output_dir=$task_name
optim="zero2"
num_vision_tower=4

master_port=9001


msrun --bind_core=True --worker_num=$device_num --local_worker_num=$device_num --master_port=$master_port --log_dir=$output_dir \
python test_modules/test_cambrian_llama.py \
    --device_target Ascend \
    --is_distribute True \
    --max_device_memory 59GB \
    --enable_fa True \
    --amp_level O2 \
    --force_param_fp16 True \
    --gradient_checkpointing True \
    --optim $optim \
    --shard_size $device_num \
    --num_vision_tower $num_vision_tower \
    \
    > .log_msrun.txt 2>&1 &
