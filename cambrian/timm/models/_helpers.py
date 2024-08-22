import mindspore as ms
from mindspore import nn

import os
import numpy as np
from typing import Any, Callable, Dict, Optional, Union


def load_checkpoint(
        model: nn.Cell,
        checkpoint_path: str,
        use_ema: bool = False,
        strict: bool = True,
        remap: bool = False,
        filter_fn: Optional[Callable] = None,
):
    if use_ema:
        raise NotImplementedError

    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return

    state_dict = ms.load_checkpoint(checkpoint_path)

    if remap:
        state_dict = remap_state_dict(state_dict, model)
    elif filter_fn:
        state_dict = filter_fn(state_dict, model)

    param_not_load, ckpt_not_load = ms.load_param_into_net(model, state_dict, strict_load=strict)

    return param_not_load, ckpt_not_load


def remap_state_dict(
        state_dict: Dict[str, Any],
        model: nn.Cell,
        allow_reshape: bool = True
):
    """ remap checkpoint by iterating over state dicts in order (ignoring original keys).
    This assumes models (and originating state dict) were created with params registered in same order.
    """
    out_dict = {}
    for (ka, va), (kb, vb) in zip(model.parameters_dict().items(), state_dict.items()):
        # assert va.numel() == vb.numel(), f'Tensor size mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed.'
        assert np.prod(va.shape) == np.prod(vb.shape), f'Tensor size mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed.'
        if va.shape != vb.shape:
            if allow_reshape:
                print(f"WARNING: timm.models._helpers.remap_state_dict: reshape parameter vb:{vb.shape} to {va.shape}")
                vb = vb.reshape(va.shape)
            else:
                assert False,  f'Tensor shape mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed.'
        out_dict[ka] = vb
    return out_dict
