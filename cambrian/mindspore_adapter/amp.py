
from mindspore.amp import _grad_scale_map, _grad_unscale_map
from mindspore.amp import StaticLossScaler as _StaticLossScaler
from mindspore.amp import DynamicLossScaler as _DynamicLossScaler


class StaticLossScaler(_StaticLossScaler):
    def scale(self, inputs):
        return _grad_scale_map(self.scale_value, inputs)

    def unscale(self, inputs):
        return _grad_unscale_map(self.scale_value, inputs)


class DynamicLossScaler(_DynamicLossScaler):
    def scale(self, inputs):
        return _grad_scale_map(self.scale_value, inputs)

    def unscale(self, inputs):
        return _grad_unscale_map(self.scale_value, inputs)
