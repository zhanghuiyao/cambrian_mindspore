import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

import math
import numpy as np
from typing import List, Tuple, Optional, Union

from cambrian.timm.layers.grid import ndgrid


def pixel_freq_bands_np(
        num_bands: int,
        max_freq: float = 224.,
        linear_bands: bool = True,
) -> np.array:
    if linear_bands:
        bands = np.linspace(1.0, max_freq / 2, num_bands, dtype=np.float32)
    else:
        bands = 2 ** np.linspace(0, math.log(max_freq, 2) - 1, num_bands, dtype=np.float32)
    return bands * np.pi


def freq_bands_np(
        num_bands: int,
        temperature: float = 10000.,
        step: int = 2,
) -> np.array:
    exp = np.arange(0, num_bands, step, dtype=np.int64).to(np.float32) / num_bands
    bands = 1. / (temperature ** exp)
    return bands


def pixel_freq_bands(
        num_bands: int,
        max_freq: float = 224.,
        linear_bands: bool = True,
) -> Tensor:
    if linear_bands:
        bands = ops.linspace(1.0, max_freq / 2, num_bands).to(ms.float32)
    else:
        bands = 2 ** ops.linspace(0, math.log(max_freq, 2) - 1, num_bands).to(ms.float32)
    return bands * np.pi


def freq_bands(
        num_bands: int,
        temperature: float = 10000.,
        step: int = 2,
) -> Tensor:
    exp = ops.arange(0, num_bands, step, dtype=ms.int32).to(ms.float32) / num_bands
    bands = 1. / (temperature ** exp)
    return bands


def build_fourier_pos_embed(
        feat_shape: List[int],
        bands: Optional[Tensor] = None,
        num_bands: int = 64,
        max_res: int = 224,
        temperature: float = 10000.,
        linear_bands: bool = False,
        include_grid: bool = False,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        dtype: ms.dtype = ms.float32,
):
    """

    Args:
        feat_shape: Feature shape for embedding.
        bands: Pre-calculated frequency bands.
        num_bands: Number of frequency bands (determines output dim).
        max_res: Maximum resolution for pixel based freq.
        temperature: Temperature for non-pixel freq.
        linear_bands: Linear band spacing for pixel based freq.
        include_grid: Include the spatial grid in output.
        in_pixels: Output in pixel freq.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(
                num_bands,
                float(max_res),
                linear_bands=linear_bands,
            )
        else:
            bands = freq_bands(
                num_bands,
                temperature=temperature,
                step=1,
            )
    else:
        if dtype is None:
            dtype = bands.dtype

    if in_pixels:
        t = [ops.linspace(-1., 1., steps=s).to(ms.float32) for s in feat_shape]
    else:
        t = [ops.arange(s, dtype=ms.int32).to(ms.float32) for s in feat_shape]

    if ref_feat_shape is not None:
        # eva's scheme for resizing rope embeddings (ref shape = pretrain)
        t = [x / f * r for x, f, r in zip(t, feat_shape, ref_feat_shape)]

    grid = ops.stack(ndgrid(t), axis=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands

    pos_sin, pos_cos = pos.sin().to(dtype), pos.cos().to(dtype)

    if include_grid:
        out = (grid, pos_sin, pos_cos)
    else:
        out = (pos_sin, pos_cos)

    return out


def build_rotary_pos_embed(
        feat_shape: List[int],
        bands: Optional[Tensor] = None,
        dim: int = 64,
        max_res: int = 224,
        temperature: float = 10000.,
        linear_bands: bool = False,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        dtype: ms.dtype = ms.float32,
):
    """

    Args:
        feat_shape: Spatial shape of the target tensor for embedding.
        bands: Optional pre-generated frequency bands
        dim: Output dimension of embedding tensor.
        max_res: Maximum resolution for pixel mode.
        temperature: Temperature (inv freq) for non-pixel mode
        linear_bands: Linearly (instead of log) spaced bands for pixel mode
        in_pixels: Pixel vs language (inv freq) mode.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // 4,
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        dtype=dtype,
    )
    num_spatial_dim = 1
    # this would be much nicer as a .numel() call to torch.Size(), but torchscript sucks
    for x in feat_shape:
        num_spatial_dim *= x

    sin_emb = sin_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    return sin_emb, cos_emb


def rot(x):
    return ops.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rot_embed(x: Tensor, sin_emb, cos_emb):
    if sin_emb.ndim == 3:
        return x * cos_emb.unsqueeze(1).expand_as(x) + rot(x) * sin_emb.unsqueeze(1).expand_as(x)
    return x * cos_emb + rot(x) * sin_emb


class RotaryEmbedding(nn.Cell):
    """ Rotary position embedding

    NOTE: This is my initial attempt at impl rotary embedding for spatial use, it has not
    been well tested, and will likely change. It will be moved to its own file.

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
            self,
            dim,
            max_res=224,
            temperature=10000,
            in_pixels=True,
            linear_bands: bool = False,
            feat_shape: Optional[List[int]] = None,
            ref_feat_shape: Optional[List[int]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape

        if feat_shape is None:
            # only cache bands
            if in_pixels:
                bands = pixel_freq_bands_np(
                    dim // 4,
                    float(max_res),
                    linear_bands=linear_bands,
                )
            else:
                bands = freq_bands_np(
                    dim // 4,
                    temperature=temperature,
                    step=1,
                )
                print(bands)

            self.bands = Parameter(Tensor(bands), requires_grad=False, name="bands")
            self.pos_embed_sin = None
            self.pos_embed_cos = None
        else:
            # cache full sin/cos embeddings if shape provided up front
            emb_sin, emb_cos = build_rotary_pos_embed(
                feat_shape=feat_shape,
                dim=dim,
                max_res=max_res,
                linear_bands=linear_bands,
                in_pixels=in_pixels,
                ref_feat_shape=self.ref_feat_shape,
            )
            self.bands = None

            self.pos_embed_sin = Parameter(emb_sin, requires_grad=False, name="pos_embed_sin")
            self.pos_embed_cos = Parameter(emb_cos, requires_grad=False, name="pos_embed_cos")

    def get_embed(self, shape: Optional[List[int]] = None):
        if self.bands is not None:
            # rebuild embeddings every call, use if target shape changes
            assert shape is not None
            return build_rotary_pos_embed(
                shape,
                self.bands,
                in_pixels=self.in_pixels,
            )
        else:
            return self.pos_embed_sin, self.pos_embed_cos

    def construct(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        sin_emb, cos_emb = self.get_embed(x.shape[2:])
        return apply_rot_embed(x, sin_emb, cos_emb)
