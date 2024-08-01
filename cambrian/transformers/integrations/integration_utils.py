import os
import tempfile
import importlib.util
from pathlib import Path

from transformers.utils import logging

from cambrian.transformers.trainer_callback import TrainerCallback
from cambrian.transformers.utils import ENV_VARS_TRUE_VALUES


logger = logging.get_logger(__name__)


# Integration functions:
def is_wandb_available():
    # any value of WANDB_DISABLED disables wandb
    if os.getenv("WANDB_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES:
        logger.warning(
            "Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the "
            "--report_to flag to control the integrations used for logging result (for instance --report_to none)."
        )
        return False
    return importlib.util.find_spec("wandb") is not None


def is_tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None or importlib.util.find_spec("tensorboardX") is not None


class TensorBoardCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).

    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, tb_writer=None):
        has_tensorboard = is_tensorboard_available()
        if not has_tensorboard:
            raise RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or"
                " install tensorboardX."
            )
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401

                self._SummaryWriter = SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter

                    self._SummaryWriter = SummaryWriter
                except ImportError:
                    self._SummaryWriter = None
        else:
            self._SummaryWriter = None
        self.tb_writer = tb_writer

    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._SummaryWriter is not None:
            self.tb_writer = self._SummaryWriter(log_dir=log_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        if self.tb_writer is None:
            self._init_summary_writer(args, log_dir)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    self.tb_writer.add_text("model_config", model_config_json)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


INTEGRATION_TO_CALLBACK = {
    # "azure_ml": AzureMLCallback,
    # "comet_ml": CometCallback,
    # "mlflow": MLflowCallback,
    # "neptune": NeptuneCallback,
    "tensorboard": TensorBoardCallback,
    # "wandb": WandbCallback,
    # "codecarbon": CodeCarbonCallback,
    # "clearml": ClearMLCallback,
    # "dagshub": DagsHubCallback,
    # "flyte": FlyteCallback,
    # "dvclive": DVCLiveCallback,
}

def get_reporting_integration_callbacks(report_to):
    for integration in report_to:
        if integration not in INTEGRATION_TO_CALLBACK:
            raise ValueError(
                f"{integration} is not supported, only {', '.join(INTEGRATION_TO_CALLBACK.keys())} are supported."
            )

    return [INTEGRATION_TO_CALLBACK[integration] for integration in report_to]
