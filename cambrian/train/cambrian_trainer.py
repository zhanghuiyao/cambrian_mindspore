import os
import dataclasses
import json
import time
import random
from typing import Dict, List, Optional, Union, Any
import numpy as np
import io
from ezcolorlog import root_logger as logger

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.communication import get_rank, get_group_size

from cambrian.transformers.trainer import (
    Trainer,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
)
from cambrian.train.dataset import LengthGroupedSampler




class CambrianTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[ms.dataset.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def training_step(self, model: nn.Cell, inputs: Dict[str, Union[Tensor, Any]]) -> Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        return loss / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model
        # if self.args.unfreeze_mm_vision_tower:
        #     opt_model.get_model().vision_tower_aux_list = nn.ModuleList(opt_model.get_vision_tower_aux_list())
        #     self.param_to_name = map_params_to_module_names([opt_model])
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            assert not (self.args.mm_projector_lr and self.args.mm_vision_sampler_lr)
            if self.args.mm_projector_lr is not None:
                projector_parameters = [n for n, _ in opt_model.parameters_and_names() if "mm_projector" in n]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.parameters_and_names() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.parameters_and_names() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.parameters_and_names() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.parameters_and_names() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            elif self.args.mm_vision_sampler_lr is not None:
                vision_sampler_parameters = [name for name, _ in opt_model.parameters_and_names() if ("vision_sampler" in name) or ("vision_query" in name) ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.parameters_and_names() if (n in decay_parameters and n not in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.parameters_and_names() if (n not in decay_parameters and n not in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.parameters_and_names() if (n in decay_parameters and n in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_vision_sampler_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.parameters_and_names() if (n not in decay_parameters and n in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_vision_sampler_lr,
                    },
                ]
            elif self.args.unfreeze_mm_vision_tower and self.args.mm_vision_tower_lr is not None:
                vision_tower_parameters = [name for name, _ in opt_model.parameters_and_names() if "vision_tower" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.parameters_and_names() if (n in decay_parameters and n not in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.parameters_and_names() if (n not in decay_parameters and n not in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.parameters_and_names() if (n in decay_parameters and n in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_vision_tower_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.parameters_and_names() if (n not in decay_parameters and n in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_vision_tower_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.parameters_and_names() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.parameters_and_names() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                raise NotImplementedError

        return self.optimizer

    def _load_optimizer_and_scheduler(self, resume_from_checkpoint):
        if resume_from_checkpoint is None:
            return

        # get path to file
        WEIGHTS_NAME = "optimizer.ckpt"
        SCHEDULER_NAME = "scheduler.ckpt"
        OPTIMIZER_PATH = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        LR_PATH = os.path.join(resume_from_checkpoint, SCHEDULER_NAME)

        if os.path.isfile(OPTIMIZER_PATH):
            optimizer_state = ms.load_checkpoint(OPTIMIZER_PATH)
            optimizer_state = optimizer_state['optimizer_state']
            ms.load_param_into_net(self.optimizer, optimizer_state)
            logger.info(f"Optimizer state successfully loaded from {OPTIMIZER_PATH}")
        else:
            logger.warning(f"Not exist optimizer state checkpoint path: `{OPTIMIZER_PATH}`")

        if os.path.isfile(LR_PATH):
            lr_scheduler_state = ms.load_checkpoint(LR_PATH)
            ms.load_param_into_net(self.lr_scheduler, lr_scheduler_state)
            logger.info(f"LR scheduler state successfully loaded from {LR_PATH}")
        else:
            logger.warning(f"Not exist lr scheduler state checkpoint path: `{LR_PATH}`")

        print("Loaded optimizer and lr scheduler state done.")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):

        if resume_from_checkpoint is None:
            return

        # Getting path to file on bucket
        WEIGHTS_NAME = "cambrian_model.ckpt"
        WEIGHTS_PATH = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)

        s_time = time.time()
        if os.path.isfile(WEIGHTS_PATH):
            state_dict = ms.load_checkpoint(WEIGHTS_PATH)
            m, u = ms.load_param_into_net(model, state_dict)

            m = [n for n in m if ("_buffer" not in n) and (".inv_freq" not in n)]
            if len(m) > 0:
                logger.warning(f"missing keys num: {len(m)}, top 10 name is: {m[:10]}")
            if len(u) > 0:
                logger.warning(f"unexpected keys num: {len(u)}, top 10 name is: {u[:10]}")

            logger.info(f"load checkpoint from `{WEIGHTS_PATH}` success, time cost: {time.time() - s_time:.2f}s")
        else:
            logger.warning(f"No available pre-trained weights")

    def _save_checkpoint(self, model, trial, metrics=None):
        from cambrian.transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        # Names of files
        TRAINING_ARGS_NAME = "training_args.ckpt"
        WEIGHTS_NAME = "cambrian_model.ckpt"
        SCHEDULER_NAME = "scheduler.ckpt"
        TRAINER_STATE_NAME = "trainer_state.json"

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # Name of files to save
        rank, world_size = get_rank(), get_group_size()
        WEIGHTS_NAME = f'weights_rank-{rank:08d}-of-{world_size:08d}-{WEIGHTS_NAME}'
        WEIGHTS_NAME_OPT = f'optimizer_rank-{rank:08d}-of-{world_size:08d}-{WEIGHTS_NAME}'

        # Path of files to save
        WEIGHTS_NAME_PATH = os.path.join(output_dir, WEIGHTS_NAME)
        WEIGHTS_NAME_OPT_PATH = os.path.join(output_dir, WEIGHTS_NAME_OPT)
        LR_PATH = os.path.join(output_dir, SCHEDULER_NAME)
        TRAIN_ARGS_PATH = os.path.join(output_dir, TRAINING_ARGS_NAME)
        TRAINER_STATE_NAME_PATH = os.path.join(output_dir, TRAINER_STATE_NAME)

        ms.save_checkpoint(model if model is not None else self.model, WEIGHTS_NAME_PATH)
        ms.save_checkpoint(self.optimizer, WEIGHTS_NAME_OPT_PATH)
        ms.save_checkpoint(self.lr_scheduler, LR_PATH)

        json_string = json.dumps(dataclasses.asdict(self.state), indent=2, sort_keys=True) + "\n"
        with open(TRAINER_STATE_NAME_PATH, 'w') as f:
            f.write(json_string)

        # TODO: save rng states
        # rng_states = {
        #     "python": random.getstate(),
        #     "numpy": np.random.get_state(),
        #     "mindspore": ms.get_rng_state(),
        # }

        logger.info(f"save checkpoint `{WEIGHTS_NAME_PATH}` success")

    def get_train_dataloader(self) -> ms.dataset.Dataset:
        loader = super().get_train_dataloader()
        return loader

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        ckpt_prefix = os.path.join(output_dir, "model_ckpt")
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        rank, world_size = get_rank(), get_group_size()
        ckpt_path = f'{ckpt_prefix}_rank-{rank:08d}-of-{world_size:08d}.ckpt'
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        ms.save_checkpoint(self.model, ckpt_path)
        print(f'checkpoint saved to {ckpt_path}\n', end='')

        # TODO
        # 1. save tokenizer
        # 2. save args
        # 3. save model.config

    """Override to add custom logs"""

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        raise NotImplementedError
