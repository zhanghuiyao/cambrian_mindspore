#!/bin/bash

# envirment
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
device_num=8

export MS_ENABLE_NUMA=0
export GLOG_v=2

export MS_MEMORY_STATISTIC=1
export MS_DEV_RUNTIME_CONF="synchronize:True"


# common setting
master_port=9001

# hyper-parameters
task_name="cambrian-8b-finetune"
model_name_or_path="your_path_to_llama3"
image_folder="your_path_to_image_folder"
pretrain_mm_mlp_adapter="./checkpoints/cambrian-8b-pretrain/mm_projector.bin"
ckpt_dir="checkpoints"
data_path="your_path_to_pretrain_jsonl e.g. Cambrian7M_withsystemprompt.jsonl"
per_device_train_batch_size=8
enable_flash_attention="True"
optim="adamw_zero2_mindspore"
adamw_enable_fuse="True"
adamw_zero_shard_size=8
output_dir=$task_name"_bs"$per_device_train_batch_size"_zero2_fuseop_shard"$adamw_zero_shard_size"_8cards"



msrun --bind_core=True --worker_num=$device_num --local_worker_num=$device_num --master_port=$master_port --log_dir=$output_dir \
python -u cambrian/train/train.py \
    --model_name_or_path $model_name_or_path \
    --version llama_v3 \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower_aux_list '["siglip/CLIP-ViT-SO400M-14-384", "openai/clip-vit-large-patch14-336", "facebook/dinov2-giant-res378", "clip-convnext-XXL-multi-stage"]' \
    --vision_tower_aux_token_len_list '[576, 576, 576, 9216]' \
    --image_token_len 576 \
    --num_query_group 1 \
    --query_num_list '[576]' \
    --connector_depth 3 \
    --image_position 91 \
    --vision_hidden_size 1024 \
    --connector_only False \
    --num_of_vision_sampler_layers 10 \
    --start_of_vision_sampler_layers 0 \
    --stride_of_vision_sampler_layers 3 \
    --mm_projector_type sva \
    --unfreeze_mm_vision_tower False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --output_dir $output_dir/$ckpt_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 4e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --run_name $task_name \
    \
    --device_target Ascend \
    --is_distribute True \
    --max_device_memory 59GB \
    --enable_flash_attention $enable_flash_attention \
    --fp16 True \
    --optim $optim \
    --adamw_enable_fuse $adamw_enable_fuse \
    --adamw_zero_shard_size $adamw_zero_shard_size \
    --save_safetensors False \
    --dataloader_num_workers 1 \
    \
    > .log_msrun.txt 2>&1 &


    # --pretrain_mm_mlp_adapter $pretrain_mm_mlp_adapter \
    # --per_device_eval_batch_size 4 \