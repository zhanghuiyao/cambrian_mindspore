#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MS_ENABLE_NUMA=0
export MS_MEMORY_STATISTIC=1
export GLOG_v=2

export MS_DEV_RUNTIME_CONF="synchronize:True"


# hyper-parameters
task_name="test_cambrian_llama_8p"
output_dir=$task_name

msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9001 --log_dir=$output_dir \
python test_modules/test_cambrian_llama.py \
    --device_target Ascend \
    --optim zero2 \
    --shard_size 8
