import mindspore as ms
from mindspore import nn
from mindspore.train.amp import AMP_BLACK_LIST, AMP_WHITE_LIST, _auto_black_list


def auto_mixed_precision(network, amp_level="O0", dtype=ms.float16):
    """
    auto mixed precision function.

    Args:
        network (Cell): Definition of the network.
        amp_level (str): Supports ["O0", "O1", "O2", "O3"]. Default: "O0".

            - "O0": Do not change.
            - "O1": Cast the operators in white_list to float16, the remaining operators are kept in float32.
            - "O2": Cast network to float16, keep operators in black_list run in float32,
            - "O3": Cast network to float16.

    Raises:
        ValueError: If amp level is not supported.

    Examples:
        >>> from mindspore import amp, nn
        >>> network = LeNet5()
        >>> amp_level = "O2"
        >>> net = amp.auto_mixed_precision(network, amp_level, dtype=ms.float16)
    """

    if not isinstance(network, nn.Cell):
        raise TypeError("The network type should be Cell.")

    if amp_level == "O0":
        pass
    elif amp_level == "O1":
        raise NotImplementedError
    elif amp_level == "O2":
        _auto_black_list(
            network,
            AMP_BLACK_LIST + [nn.GroupNorm, nn.SiLU],
            dtype,
        )
    elif amp_level == "O3":
        network.to_float(dtype)
    else:
        raise ValueError("The amp level {} is not supported".format(amp_level))
    return network
