import time
import warnings
from typing import Optional, Union, List
from abc import ABC

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops, Tensor

from transformers import logging


logger = logging.get_logger(__name__)


class StoppingCriteriaList(list):
    def __call__(self, input_ids: Tensor, scores: Tensor, **kwargs) -> Tensor:
        is_done = ops.full((input_ids.shape[0],), False, dtype=ms.bool_)
        for criteria in self:
            is_done = ops.logical_or(is_done, criteria(input_ids, scores, **kwargs))
        return is_done

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
            elif isinstance(stopping_criterium, MaxNewTokensCriteria):
                return stopping_criterium.max_length
        return None


class StoppingCriteria(ABC):
    """Abstract base class for all stopping criteria that can be applied during generation.

    If your stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True,
    output_scores=True` to `generate`.
    """

    def __call__(self, input_ids: Tensor, scores: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class MaxLengthCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.

    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of tokens.
        max_position_embeddings (`int`, *optional*):
            The maximum model length, as defined by the model's `config.max_position_embeddings` attribute.
    """

    def __init__(self, max_length: int, max_position_embeddings: Optional[int] = None):
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings

    def __call__(self, input_ids: Tensor, scores: Tensor, **kwargs) -> Tensor:
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
            logger.warning_once(
                "This is a friendly reminder - the current text generation call will exceed the model's predefined "
                f"maximum length ({self.max_position_embeddings}). Depending on the model, you may observe "
                "exceptions, performance degradation, or nothing at all."
            )
        return ops.full((input_ids.shape[0],), is_done, dtype=ms.bool_)


class MaxNewTokensCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the generated number of tokens exceeds `max_new_tokens`. Keep in
    mind for decoder-only type of transformers, this will **not** include the initial prompted tokens. This is very
    close to `MaxLengthCriteria` but ignores the number of initial tokens.

    Args:
        start_length (`int`):
            The number of initial tokens.
        max_new_tokens (`int`):
            The maximum number of tokens to generate.
    """

    def __init__(self, start_length: int, max_new_tokens: int):
        warnings.warn(
            "The class `MaxNewTokensCriteria` is deprecated and will be removed in v4.43. "
            f"Please use `MaxLengthCriteria(max_length={start_length + max_new_tokens})` "
            "with `max_length = start_length + max_new_tokens` instead.",
            FutureWarning,
        )
        self.start_length = start_length
        self.max_new_tokens = max_new_tokens
        self.max_length = start_length + max_new_tokens

    def __call__(self, input_ids: Tensor, scores: Tensor, **kwargs) -> Tensor:
        is_done = input_ids.shape[-1] >= self.max_length
        return ops.full((input_ids.shape[0],), is_done, dtype=ms.bool_)


class MaxTimeCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the
    time will start being counted when you initialize this function. You can override this by passing an
    `initial_time`.

    Args:
        max_time (`float`):
            The maximum allowed time in seconds for the generation.
        initial_time (`float`, *optional*, defaults to `time.time()`):
            The start of the generation allowed time.
    """

    def __init__(self, max_time: float, initial_timestamp: Optional[float] = None):
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    def __call__(self, input_ids: Tensor, scores: Tensor, **kwargs) -> Tensor:
        is_done = time.time() - self.initial_timestamp > self.max_time
        return ops.full((input_ids.shape[0],), is_done, dtype=ms.bool_)


class EosTokenCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the "end-of-sequence" token is generated.
    By default, it uses the `model.generation_config.eos_token_id`.

    Args:
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
    """

    def __init__(self, eos_token_id: Union[int, List[int], Tensor]):
        if not isinstance(eos_token_id, Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = Tensor(eos_token_id)
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: Tensor, scores: Tensor, **kwargs) -> Tensor:
        is_done = mnp.isin(input_ids[:, -1], self.eos_token_id)
        return is_done
