import os
import re
import copy
import json
import logging
import pathlib
import numpy as np
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from PIL import Image
from ezcolorlog import root_logger as logger
from packaging import version

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.communication.management import init, get_rank, get_group_size

import tokenizers
from transformers import PreTrainedTokenizer, AutoTokenizer

from cambrian.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)
from cambrian import conversation as conversation_lib

from cambrian.train.cambrian_trainer import CambrianTrainer
from cambrian.train.dataset import DataArguments, make_supervised_data_module

from cambrian.mm_utils import tokenizer_image_token, tokenizer_image_token_llama3
from cambrian.model import CambrianLlamaForCausalLM

from cambrian.transformers import TrainingArguments as _TrainingArguments
from cambrian.transformers import (
    PreTrainedModel,
    HfArgumentParser,
    LlamaForCausalLM
)

from cambrian.mindspore_adapter.training_args import init_environment, MindSporeArguments
from cambrian.mindspore_adapter.utils import _is_parallel


logger.setLevel(logging.WARNING)

local_rank = 0

PRINT_LOGS = True


def print_rank0(*args):
    if local_rank in (0, -1) and PRINT_LOGS:
        print(*args)


def log_rank0(log):
    if local_rank in (0, -1) and PRINT_LOGS:
        logger.info(log, stacklevel=2)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_aux_list: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    vision_tower_aux_token_len_list: Optional[str] = field(default=None)
    image_token_len: Optional[int] = field(default=576)
    num_query_group: Optional[int] = field(default=1)
    query_num_list: Optional[str] = field(default='[576]')
    connector_depth: Optional[int] = field(default=1)
    vision_hidden_size: Optional[int] = field(default=1024)
    connector_only: bool = field(default=True)
    num_of_vision_sampler_layers: Optional[int] = field(default=10)
    start_of_vision_sampler_layers: Optional[int] = field(default=16)
    stride_of_vision_sampler_layers: Optional[int] = field(default=1)


@dataclass
class TrainingArguments(MindSporeArguments, _TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_mindspore")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_sampler_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    mm_vision_tower_lr: Optional[float] = None

    # sanity check arg
    batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "The total batch size for training. If passed, will be used to check that the "
                          "`per_device_train_batch_size` is set correctly."}
    )

    train_continue: bool = False
    resume_from_checkpoint: Optional[str] = ""


def train():
    global local_rank

    log_rank0(f"Training starting...")

    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        TrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    init_environment(training_args)

    local_rank = training_args.local_rank
    compute_dtype = (ms.float16 if training_args.fp16 else (ms.bfloat16 if training_args.bf16 else ms.float32))

    # verify that the train_batch_size is set correctly
    if training_args.batch_size is not None:
        world_size = get_group_size() if _is_parallel() else 1

        if training_args.per_device_train_batch_size is None:
            raise ValueError("If train_batch_size is set, per_device_train_batch_size must be set")

        if training_args.batch_size != training_args.per_device_train_batch_size * world_size:
            raise ValueError(
                f"train_batch_size ({training_args.train_batch_size}) must equal per_device_train_batch_size ({training_args.per_device_train_batch_size}) * world_size ({world_size})")

        logger.warning(
            f"per_device_train_batch_size is correctly set to {training_args.per_device_train_batch_size} with world_size {world_size} to match train_batch_size {training_args.batch_size}")
        logger.warning(f"train_batch_size is {training_args.train_batch_size}")

    use_cohere = False
    data_args.image_token_len = model_args.image_token_len

    if model_args.vision_tower_aux_list is not None:
        # copy image_token_len and image_position to model_args
        # data_args.image_token_len = model_args.image_token_len
        model_args.image_position = data_args.image_position

        # Assuming model_args.model_name_or_path is a string that includes the model size
        model_name = model_args.model_name_or_path

        # Regular expression to find the number of parameters in the model's name (assuming a convention like 'ModelName-30b')
        match = re.search(r'(\d+)b', model_name)
        num_parameters_billion = float(match.group(1)) if match else 0

        # Determine if bfloat16 should be used based on the model's size
        use_bfloat16 = training_args.bf16 or num_parameters_billion > 30

        if "yi" in model_args.model_name_or_path.lower():
            raise NotImplementedError
        elif "mistral" in model_name.lower():
            raise NotImplementedError
        elif "phi-3" in model_name.lower():
            raise NotImplementedError
        else:
            logger.warning(f"Vision tower, loading CambrianLlamaForCausalLM: {model_args.model_name_or_path}")
            model = CambrianLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                mindspore_dtype=(ms.bfloat16 if use_bfloat16 else None),
                # cache_dir=training_args.cache_dir,
                # do_sample=True,
            )
            model.generation_config.do_sample = True  # FIXME: move to `.from_pretrain()`
    else:
        logger.warning(f"No vision tower, loading pure language model: {model_args.model_name_or_path}")
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            # cache_dir=training_args.cache_dir,
        )
    model.config.use_cache = False
    model.generation_config.do_sample = True

    if model_args.freeze_backbone:
        model.model.requires_grad = False

    log_rank0("Building model done.")

    if training_args.bits in [4, 8]:
        raise NotImplementedError

    if training_args.gradient_checkpointing:
        log_rank0("Using gradient checkpointing")

    if training_args.lora_enable:
        raise NotImplementedError

    log_rank0("Configuring tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        raise NotImplementedError
    elif model_args.version == "v0.5":
        raise NotImplementedError
    elif model_args.version == "llama_v3":
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token_id = 128002
    else:
        tokenizer.pad_token = tokenizer.unk_token

    # FIXME: level 0, The official implementation will not modify `default_comversation` in llama_v3
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        logger.warning(f"Conversation version {model_args.version} not found. Using default `vicuna_v1`")
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if use_cohere:
        tokenizer.pad_token_id = 0
        print_rank0("tokenizer id is", tokenizer.pad_token_id)

    if model_args.vision_tower_aux_list is not None:
        model_args.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        model_args.vision_tower_aux_list = json.loads(model_args.vision_tower_aux_list)
        model_args.vision_tower_aux_token_len_list = json.loads(model_args.vision_tower_aux_token_len_list)
        model_args.query_num_list = json.loads(model_args.query_num_list)

        model.get_model().initialize_vision_modules(
            model_args=model_args,
        )
        model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower

        vision_tower_aux_list = None
        if model_args.vision_tower_aux_list is not None:
            vision_tower_aux_list = model.get_vision_tower_aux_list()

        if vision_tower_aux_list is not None:
            data_args.image_processor_aux_list = [vision_tower_aux.image_processor
                                                  for vision_tower_aux in vision_tower_aux_list]
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.image_position = data_args.image_position

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad = False
            tune_modules = ['mm_projector', 'pos_emb', 'vision_sampler', 'vision_sampler_layers', 'vision_query',
                            'image_newline']
            for name, param in model.parameters_and_names():
                if any(listed_name in name for listed_name in tune_modules):
                    print_rank0('tuning {}'.format(name))
                    param.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.get_parameters():
                p.requires_grad = False
        if training_args.unfreeze_mm_vision_tower:
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    for p in vision_tower_aux.parameters():
                        p.requires_grad = True

        if training_args.bits in [4, 8]:
            raise NotImplementedError

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.image_token_len = data_args.image_token_len = model_args.image_token_len
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_sampler_lr = training_args.mm_vision_sampler_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.vision_tower_aux_token_len_list = data_args.vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list
        model.config.image_token_len = data_args.image_token_len
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        raise NotImplementedError

    log_rank0("Configuring data module...")
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    if "wandb" in training_args.report_to:
        raise NotImplementedError

    log_rank0("Configuring trainer...")
    trainer = CambrianTrainer(model=model,
                              tokenizer=tokenizer,
                              args=training_args,
                              **data_module)
    if training_args.train_continue:
        resume_from_checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    log_rank0(f"Training finished: {training_args.output_dir}")

    trainer.save_state()

    model.config.use_cache = True

    log_rank0("Saving model...")
    if training_args.lora_enable:
        raise NotImplementedError
    else:
        pass
        # FIXME: level 1, save trainer
        # safe_save_model_for_hf_trainer(trainer=trainer,
        #                                output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
