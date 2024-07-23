from mindspore import nn


class HardMish(nn.Cell):
    def __init__(self):
        super(HardMish, self).__init__()

    def construct(self, x):
        return 0.5 * x * (x + 2).clamp(min=0, max=2)
