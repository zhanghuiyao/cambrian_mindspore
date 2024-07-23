from typing import Any, Optional


class set_layer_config:
    """ Layer config context manager that allows setting all layer config flags at once.
    If a flag arg is None, it will not change the current value.
    """
    def __init__(
            self,
            scriptable: Optional[bool] = None,
            exportable: Optional[bool] = None,
            no_jit: Optional[bool] = None,
            no_activation_jit: Optional[bool] = None):
        global _SCRIPTABLE
        global _EXPORTABLE
        global _NO_JIT
        global _NO_ACTIVATION_JIT
        self.prev = _SCRIPTABLE, _EXPORTABLE, _NO_JIT, _NO_ACTIVATION_JIT
        if scriptable is not None:
            _SCRIPTABLE = scriptable
        if exportable is not None:
            _EXPORTABLE = exportable
        if no_jit is not None:
            _NO_JIT = no_jit
        if no_activation_jit is not None:
            _NO_ACTIVATION_JIT = no_activation_jit

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> bool:
        global _SCRIPTABLE
        global _EXPORTABLE
        global _NO_JIT
        global _NO_ACTIVATION_JIT
        _SCRIPTABLE, _EXPORTABLE, _NO_JIT, _NO_ACTIVATION_JIT = self.prev
        return False
