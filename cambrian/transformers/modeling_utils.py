import copy
from typing import Optional, Dict, Union, Any
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from transformers import PretrainedConfig, GenerationConfig
from transformers.utils import logging

from cambrian.transformers.generation.utils import GenerationMixin
from cambrian.transformers.utils.import_utils import is_flash_attn_2_available

logger = logging.get_logger(__name__)


class ModuleUtilsMixin:

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.embedding_table" for name, cell in self.cells_and_names() if isinstance(cell, nn.Embedding)
            ]
            total_parameters = [
                parameter for name, parameter in self.parameters_and_names() if name not in embedding_param_names
            ]
        else:
            total_parameters = list([p for p in self.get_parameters()])

        total_numel = []
        is_loaded_in_4bit = getattr(self, "is_loaded_in_4bit", False)

        if is_loaded_in_4bit:
            raise NotImplementedError(
                "4 bit not support on MindSpore 2.3"
            )

        for param in total_parameters:
            if param.requires_grad or not only_trainable:
                # For 4bit models, we need to multiply the number of parameters by 2 as half of the parameters are
                # used for the 4bit quantization (uint8 tensors are stored)
                if is_loaded_in_4bit:
                    raise NotImplementedError("4 bit not support on MindSpore 2.3")
                else:
                    total_numel.append(param.numel())

        return sum(total_numel)

    def estimate_tokens(self, input_dict: Dict[str, Union[Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (`dict`): The model inputs.

        Returns:
            `int`: The total number of tokens.
        """
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        if self.main_input_name in input_dict:
            return input_dict[self.main_input_name].numel()
        elif "estimate_tokens" not in self.warnings_issued:
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            self.warnings_issued["estimate_tokens"] = True
        return 0

    def floating_point_ops(
        self, input_dict: Dict[str, Union[Tensor, Any]], exclude_embeddings: bool = True
    ) -> int:
        """
        Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
        batch with this transformer model. Default approximation neglects the quadratic dependency on the number of
        tokens (valid if `12 * d_model << sequence_length`) as laid out in [this
        paper](https://arxiv.org/pdf/2001.08361.pdf) section 2.1. Should be overridden for transformers with parameter
        re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

        Args:
            batch_size (`int`):
                The batch size for the forward pass.

            sequence_length (`int`):
                The number of tokens in each line of the batch.

            exclude_embeddings (`bool`, *optional*, defaults to `True`):
                Whether or not to count embedding and softmax operations.

        Returns:
            `int`: The number of floating-point operations.
        """

        return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)


class PreTrainedModel(nn.Cell, ModuleUtilsMixin, GenerationMixin):
    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    model_tags = None

    _auto_class = None
    _no_split_modules = None
    _skip_keys_device_placement = None
    _keep_in_fp32_modules = None

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None
    # a list of `state_dict` keys that are potentially tied to another key in the state_dict.
    _tied_weights_keys = None

    is_parallelizable = False
    supports_gradient_checkpointing = False
    _is_stateful = False

    # Flash Attention 2 support
    _supports_flash_attn_2 = False

    # SDPA support
    _supports_sdpa = False

    # Has support for a `Cache` instance as `past_key_values`? Does it support a `StaticCache`?
    _supports_cache_class = False
    _supports_static_cache = False

    # Has support for a `QuantoQuantizedCache` instance as `past_key_values`
    _supports_quantized_cache = False

    def __init__(self, config):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )

        self.config = config
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)

        config, _ = cls.config_class.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=None,
            return_unused_kwargs=True,
            force_download=False,
            resume_download=False,
            proxies=None,
            local_files_only=False,
            token=None,
            revision="main",
            subfolder="",
            _from_auto=False,
            _from_pipeline=None,
            **kwargs
        )

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        config = cls._autoset_attn_implementation(
            config,
            use_flash_attention_2=use_flash_attention_2
        )

        model = cls(config)

        # If it is a model with generation capabilities, attempt to load the generation config
        if model.can_generate() and pretrained_model_name_or_path is not None:
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=None,
                    force_download=False,
                    resume_download=False,
                    proxies=None,
                    local_files_only=False,
                    token=None,
                    revision="main",
                    subfolder="",
                    _from_auto=False,
                    _from_pipeline=None,
                )
            except OSError:
                logger.info(
                    "Generation config file not found, using a generation config created from the model config."
                )
                pass

        return model

    @classmethod
    def _autoset_attn_implementation(
            cls,
            config,
            use_flash_attention_2: bool = False,
    ):
        """
        Automatically checks and dispatches to a default attention implementation. In order of priority:
            1. An implementation specified in `config._attn_implementation` (due for example to the argument attn_implementation="sdpa" in from_pretrained).
            2. DEPRECATED: if use_flash_attention_2 is set to `True` and `flash_attn` is available, flash attention. (`LlamaFlashAttention` for example)
            3. SDPA implementation, if available and supported by the model type. (`LlamaSdpaAttention` for example)
            4. The default model's implementation otherwise (`LlamaAttention` for example) .
        """
        # Here we use config._attn_implementation_internal to check whether the attention implementation was explicitely set by the user.
        # The property `PretrainedConfig._attn_implementation` is never `None`, for backward compatibility (always fall back on "eager").
        # The `hasattr` here is used as some Transformers tests for some reason do not call PretrainedConfig __init__ (e.g. test_no_super_init_config_and_model)
        requested_attn_implementation = None
        if hasattr(config, "_attn_implementation_internal") and config._attn_implementation_internal is not None:
            if config._attn_implementation != "flash_attention_2" and use_flash_attention_2:
                raise ValueError(
                    f'Both attn_implementation="{config._attn_implementation}" and `use_flash_attention_2=True` were used when loading the model, which are not compatible.'
                    ' We recommend to just use `attn_implementation="flash_attention_2"` when loading the model.'
                )

            if config._attn_implementation not in ["eager", "sdpa", "flash_attention_2"]:
                message = f'Specified `attn_implementation="{config._attn_implementation}"` is not supported. The only possible arguments are `attn_implementation="eager"` (manual attention implementation)'
                if cls._supports_flash_attn_2:
                    message += ', `"attn_implementation=flash_attention_2"` (implementation using flash attention 2)'
                if cls._supports_sdpa:
                    message += ', `"attn_implementation=sdpa"` (implementation using torch.nn.functional.scaled_dot_product_attention)'
                raise ValueError(message + ".")

            # If a config is passed with a preset attn_implementation, we skip the automatic dispatch and use the user-provided config, with hard checks that the requested attention implementation is available.
            requested_attn_implementation = config._attn_implementation_internal

        if use_flash_attention_2:
            logger.warning_once(
                'The model was loaded with use_flash_attention_2=True, which is deprecated and may be removed in a future release. Please use `attn_implementation="flash_attention_2"` instead.'
            )
            config._attn_implementation = "flash_attention_2"

        if config._attn_implementation == "flash_attention_2":
            cls._check_and_enable_flash_attn_2(
                config,
                hard_check_only=False,
            )
        else:
            config._attn_implementation = "eager"

        return config

    def _init_weights(self, module):
        """
        Initialize the weights. This method should be overridden by derived class and is
        the only initialization method that will be called when loading a checkpoint
        using `from_pretrained`. Any attempt to initialize outside of this function
        will be useless as the init function are all replaced with skip.
        """
        pass

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)
        new_embeddings.requires_grad = old_embeddings.requires_grad
        new_embeddings.embedding_table.name = old_embeddings.embedding_table.name
        self.set_input_embeddings(new_embeddings)
        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None

        # Update new_num_tokens with the actual size of new_embeddings
        if pad_to_multiple_of is not None:
            new_num_tokens = new_embeddings.embedding_table.shape[0]

        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head = self.get_output_embeddings()
            if isinstance(old_lm_head, nn.Embedding):
                new_lm_head = self._get_resized_embeddings(old_lm_head, new_num_tokens)
            else:
                new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            new_lm_head.requires_grad = old_lm_head.requires_grad
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()


    def resize_token_embeddings(
            self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `mindspore.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        if hasattr(self.config, "text_config"):
            self.config.text_config.vocab_size = model_embeds.embedding_table.shape[0]
        # TODO: to be removed after v4.42, config.vocab_size is deprecated for models that have a config.text_config

        self.config.vocab_size = model_embeds.embedding_table.shape[0]
        self.vocab_size = model_embeds.embedding_table.shape[0]

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        if getattr(self.config, "tie_word_embeddings", True):
            raise NotImplementedError

        if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
            raise NotImplementedError

        for name, module in self.cells_and_names():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc


        Return:
            `torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            `new_num_tokens` is `None`
        """

        if pad_to_multiple_of is not None:
            if not isinstance(pad_to_multiple_of, int):
                raise ValueError(
                    f"Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, which is not and integer. Please make sure to pass an integer"
                )
            if new_num_tokens is None:
                new_num_tokens = old_embeddings.embedding_table.shape[0]
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        else:
            logger.info(
                "You are resizing the embedding layer without providing a `pad_to_multiple_of` parameter. This means that the new embedding"
                f" dimension will be {new_num_tokens}. This might induce some performance reduction as *Tensor Cores* will not be available."
                " For more details about this, or help on choosing the correct value for resizing, refer to this guide:"
                " https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc"
            )

        if new_num_tokens is None:
            return old_embeddings

        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        old_num_tokens, old_embedding_dim = old_embeddings.embedding_table.shape

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
            dtype=old_embeddings.embedding_table.dtype,
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)

        new_embeddings_np = new_embeddings.embedding_table.data.asnumpy()
        new_embeddings_np[:n, :] = old_embeddings.embedding_table.data.asnumpy()[:n, :]

        new_embeddings.embedding_table.set_data(Tensor(new_embeddings_np))

        return new_embeddings

    def get_input_embeddings(self) -> nn.Cell:
        """
        Returns the model's input embeddings.

        Returns:
            `nn.Cell`: A torch module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def set_input_embeddings(self, value: nn.Cell):
        """
        Set model's input embeddings.

        Args:
            value (`nn.Cell`): A module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def get_output_embeddings(self) -> nn.Cell:
        """
        Returns the model's output embeddings.

        Returns:
            `nn.Cell`: A torch module mapping hidden states to vocabulary.
        """
        return None  # Overwrite for models with output embeddings

    @classmethod
    def _check_and_enable_flash_attn_2(
            cls,
            config,
            hard_check_only: bool = False,
    ) -> PretrainedConfig:
        """
        Checks the availability of Flash Attention 2 and compatibility with the current model.

        If all checks pass and `hard_check_only` is False, the method will set the config attribute `attn_implementation` to "flash_attention_2" so that the model can initialize the correct attention module.
        """
        if not cls._supports_flash_attn_2:
            raise ValueError(
                f"{cls.__name__} does not support Flash Attention 2.0 yet. Please request to add support where"
                f" the model is hosted, on its model hub page: https://huggingface.co/{config._name_or_path}/discussions/new"
                " or in the Transformers GitHub repo: https://github.com/huggingface/transformers/issues/new"
            )

        if not is_flash_attn_2_available():
            raise ImportError(
                f"FlashAttention2 has been toggled on, but it cannot be used due to some error; "
                f"Please refer to the README.md to install MindSpore and CANN on Ascend device."
            )

        _is_bettertransformer = getattr(cls, "use_bettertransformer", False)

        if _is_bettertransformer:
            raise ValueError(
                "Flash Attention 2 and BetterTransformer API are not compatible. Please make sure to disable BetterTransformers by doing model.reverse_bettertransformer()"
            )

        if not hard_check_only:
            config._attn_implementation = "flash_attention_2"
        return config

    def _get_resized_lm_head(
        self, old_lm_head: nn.Dense, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False
    ) -> nn.Dense:
        """
        Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_lm_head (`torch.nn.Linear`):
                Old lm head liner layer to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Linear` module of the model without doing anything. transposed (`bool`, *optional*, defaults
                to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.shape` is `lm_head_dim,
                vocab_size` else `vocab_size, lm_head_dim`.

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
            `None`
        """
        if new_num_tokens is None:
            return old_lm_head

        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        old_num_tokens, old_lm_head_dim = (
            old_lm_head.weight.shape if not transposed else old_lm_head.weight.t().shape()
        )

        if old_num_tokens == new_num_tokens:
            return old_lm_head

        if not isinstance(old_lm_head, nn.Dense):
            raise TypeError(
                f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Dense}. You"
                " should either use a different resize function or make sure that `old_lm_head` are an instance of"
                f" {nn.Dense}."
            )

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_lm_head = nn.Dense(
            *new_lm_head_shape,
            has_bias=has_new_lm_head_bias,
            dtype=old_lm_head.weight.dtype,
        )

        # initialize new lm head (in particular added tokens)
        self._init_weights(new_lm_head)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        self._copy_lm_head_original_to_resized(
            new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
        )

        return new_lm_head

    def _copy_lm_head_original_to_resized(
        self, new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
    ):
        new_weight_np = new_lm_head.weight.data.asnumpy()
        old_weight_np = old_lm_head.weight.data.asnumpy()

        # Copy old lm head weights to new lm head
        if not transposed:
            new_weight_np[:num_tokens_to_copy, :] = old_weight_np[:num_tokens_to_copy, :]
        else:
            new_weight_np[:, :num_tokens_to_copy] = old_weight_np[:, :num_tokens_to_copy]
        new_lm_head.weight.set_data(Tensor(new_weight_np))

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_bias_np = new_lm_head.bias.data.asnumpy()
            old_bias_np = old_lm_head.bias.data.asnumpy()

            new_bias_np[:num_tokens_to_copy] = old_bias_np[:num_tokens_to_copy]

            new_lm_head.bias.set_data(Tensor(new_bias_np))

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation.
        # Alternativelly, the model can also have a custom `generate` function.
        if "GenerationMixin" in str(cls.prepare_inputs_for_generation) and "GenerationMixin" in str(cls.generate):
            return False
        return True


    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        raise NotImplementedError
