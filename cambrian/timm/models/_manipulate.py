import collections.abc
import math
import re
from collections import defaultdict
from itertools import chain
from typing import Iterator, Tuple, Any, Union, Dict, Callable

import mindspore as ms
from mindspore import nn


def named_apply(
        fn: Callable,
        module: nn.Cell, name='',
        depth_first: bool = True,
        include_root: bool = False,
) -> nn.Cell:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.cells_and_names():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def named_modules_with_params(
        module: nn.Cell,
        name: str = '',
        depth_first: bool = True,
        include_root: bool = False,
):
    if module.parameters_dict() and not depth_first and include_root:
        yield name, module
    for child_name, child_module in module.cells_and_names():
        child_name = '.'.join((name, child_name)) if name else child_name
        yield from named_modules_with_params(
            module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if module.parameters_dict() and depth_first and include_root:
        yield name, module


MATCH_PREV_GROUP = (99999,)


def group_with_matcher(
        named_objects: Iterator[Tuple[str, Any]],
        group_matcher: Union[Dict, Callable],
        return_values: bool = False,
        reverse: bool = False
):
    if isinstance(group_matcher, dict):
        # dictionary matcher contains a dict of raw-string regex expr that must be compiled
        compiled = []
        for group_ordinal, (group_name, mspec) in enumerate(group_matcher.items()):
            if mspec is None:
                continue
            # map all matching specifications into 3-tuple (compiled re, prefix, suffix)
            if isinstance(mspec, (tuple, list)):
                # multi-entry match specifications require each sub-spec to be a 2-tuple (re, suffix)
                for sspec in mspec:
                    compiled += [(re.compile(sspec[0]), (group_ordinal,), sspec[1])]
            else:
                compiled += [(re.compile(mspec), (group_ordinal,), None)]
        group_matcher = compiled

    def _get_grouping(name):
        if isinstance(group_matcher, (list, tuple)):
            for match_fn, prefix, suffix in group_matcher:
                r = match_fn.match(name)
                if r:
                    parts = (prefix, r.groups(), suffix)
                    # map all tuple elem to int for numeric sort, filter out None entries
                    return tuple(map(float, chain.from_iterable(filter(None, parts))))
            return float('inf'),  # un-matched layers (neck, head) mapped to largest ordinal
        else:
            ord = group_matcher(name)
            if not isinstance(ord, collections.abc.Iterable):
                return ord,
            return tuple(ord)

    # map layers into groups via ordinals (ints or tuples of ints) from matcher
    grouping = defaultdict(list)
    for k, v in named_objects:
        grouping[_get_grouping(k)].append(v if return_values else k)

    # remap to integers
    layer_id_to_param = defaultdict(list)
    lid = -1
    for k in sorted(filter(lambda x: x is not None, grouping.keys())):
        if lid < 0 or k[-1] != MATCH_PREV_GROUP[0]:
            lid += 1
        layer_id_to_param[lid].extend(grouping[k])

    if reverse:
        assert not return_values, "reverse mapping only sensible for name output"
        # output reverse mapping
        param_to_layer_id = {}
        for lid, lm in layer_id_to_param.items():
            for n in lm:
                param_to_layer_id[n] = lid
        return param_to_layer_id

    return layer_id_to_param


def group_parameters(
        module: nn.Cell,
        group_matcher,
        return_values: bool = False,
        reverse: bool = False,
):
    return group_with_matcher(
        module.parameters_and_names(), group_matcher, return_values=return_values, reverse=reverse)


def group_modules(
        module: nn.Cell,
        group_matcher,
        return_values: bool = False,
        reverse: bool = False,
):
    return group_with_matcher(
        named_modules_with_params(module), group_matcher, return_values=return_values, reverse=reverse)


def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.SequentialCell(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, nn.SequentialCell):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = run_function(start, end, functions)(x)
    if skip_last:
        x = run_function(end + 1, len(functions) - 1, functions)(x)
    return x


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.to(ms.float32)  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(axis=2, keepdims=False)
        else:
            conv_weight = conv_weight.sum(axis=1, keepdims=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.tile((1, repeat, 1, 1))[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight
