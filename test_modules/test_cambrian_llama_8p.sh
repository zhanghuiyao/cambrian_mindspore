#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
device_num=8

export MS_ENABLE_NUMA=0
export MS_MEMORY_STATISTIC=1
export GLOG_v=2

export MS_DEV_RUNTIME_CONF="synchronize:True"


# hyper-parameters
task_name="logs_test_cambrian_llama_8p"
output_dir=$task_name

master_port=9012


msrun --bind_core=True --worker_num=$device_num --local_worker_num=$device_num --master_port=$master_port --log_dir=$output_dir \
python test_modules/test_cambrian_llama.py \
    --device_target Ascend \
    --is_distribute True \
    --max_device_memory 59GB \
    --optim zero2 \
    --shard_size $device_num \
    \
    > .log_msrun.txt 2>&1 &