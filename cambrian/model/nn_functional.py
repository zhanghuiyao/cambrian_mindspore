import mindspore as ms
from mindspore import ops


def scaled_dot_product_attention(query, key, value, attn_mask=None, dtype=None):
    # force dtype(fp16 or bf16) precision calculation
    ori_dtype = query.dtype
    if dtype is not None:
        query, key, value = query.astype(dtype), key.astype(dtype), value.astype(dtype)

    if attn_mask is not None:
        attn_mask = attn_mask.masked_fill(not attn_mask, -1e5) if attn_mask.dtype == ms.bool_ else attn_mask
        attn_weight = ops.softmax(
            ops.cast(ops.matmul(query, key.swapaxes(-2, -1)) / (query.shape[-1] ** 0.5) + attn_mask, ms.float32),
            axis=-1,
        ).astype(query.dtype)
    else:
        attn_weight = ops.softmax(
            ops.cast(ops.matmul(query, key.swapaxes(-2, -1)) / (query.shape[-1] ** 0.5), ms.float32), axis=-1
        ).astype(query.dtype)

    out = ops.matmul(attn_weight, value)
    out = out.astype(ori_dtype)

    return out
