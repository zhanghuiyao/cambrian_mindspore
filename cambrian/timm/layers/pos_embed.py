from typing import List, Tuple, Optional, Union

import mindspore as ms
from mindspore import ops


def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int((num_pos_tokens - num_prefix_tokens) ** 0.5)
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.to(ms.float32)  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    # FIXME: `antialias` not support on `mindspore.ops.interpolate`
    posemb = ops.interpolate(posemb, size=new_size, mode=interpolation)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = ops.cat([posemb_prefix, posemb], axis=1)

    return posemb
