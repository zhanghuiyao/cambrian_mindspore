#!/bin/bash

#export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export MS_ENABLE_NUMA=0
#export MS_MEMORY_STATISTIC=1
export MS_DEV_RUNTIME_CONF="synchronize:True"
export GLOG_v=2


# hyper-parameters
data_path="/Users/zhanghuiyao/Desktop/cambrian_mindspore/demo/toy-dataset/alignment_2.5m.jsonl"  #  e.g. Cambrian7M_withsystemprompt.jsonl
model_name_or_path="/Users/zhanghuiyao/Desktop/cambrian_mindspore/cambrian/hf-configs/nyu-visionx-cambrian-8b"
image_folder="/Users/zhanghuiyao/Desktop/cambrian_mindspore/demo/toy-dataset/images_from_coco"
enable_flash_attention="False"
per_device_train_batch_size=1

optim="adamw_zero2_mindspore"
adamw_zero_shard_size=8

task_name="cambrian-8b-finetune"
pretrain_mm_mlp_adapter="./checkpoints/cambrian-8b-pretrain/mm_projector.bin"
ckpt_dir="checkpoints"
output_dir=$task_name"_FA-"$enable_flash_attention"_bs-"$per_device_train_batch_size

python cambrian/train/train.py \
    --model_name_or_path $model_name_or_path \
    --version llama_v3 \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower_aux_list '["siglip/CLIP-ViT-SO400M-14-384", "openai/clip-vit-large-patch14-336", "facebook/dinov2-giant-res378", "clip-convnext-XXL-multi-stage"]' \
    --vision_tower_aux_token_len_list '[576]' \
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
    --output_dir $output_dir \
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
    --is_distribute False \
    --optim $optim \
    --adamw_zero_shard_size $adamw_zero_shard_size \
    \
    --save_safetensors False \
    --device_target CPU \
    --dataloader_num_workers 1 \

    # --pretrain_mm_mlp_adapter $pretrain_mm_mlp_adapter \
    #--per_device_eval_batch_size 4 \