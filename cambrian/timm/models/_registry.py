import sys
import warnings
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Sequence, Union, Tuple
from dataclasses import replace

from cambrian.timm.models._pretrained import PretrainedCfg, DefaultCfg


_model_entrypoints: Dict[str, Callable[..., Any]] = {}  # mapping of model names to architecture entrypoint fns
_model_has_pretrained: Set[str] = set()  # set of model names that have pretrained weight url present
_model_default_cfgs: Dict[str, PretrainedCfg] = {}  # central repo for model arch -> default cfg objects
_model_pretrained_cfgs: Dict[str, PretrainedCfg] = {}  # central repo for model arch.tag -> pretrained cfgs
_model_with_tags: Dict[str, List[str]] = defaultdict(list)  # shortcut to map each model arch to all model + tag names


def split_model_name_tag(model_name: str, no_tag: str = '') -> Tuple[str, str]:
    model_name, *tag_list = model_name.split('.', 1)
    tag = tag_list[0] if tag_list else no_tag
    return model_name, tag


def get_arch_name(model_name: str) -> str:
    return split_model_name_tag(model_name)[0]


def is_model(model_name: str) -> bool:
    """ Check if a model name exists
    """
    arch_name = get_arch_name(model_name)
    return arch_name in _model_entrypoints


def model_entrypoint(model_name: str, module_filter: Optional[str] = None) -> Callable[..., Any]:
    """Fetch a model entrypoint for specified model name
    """
    arch_name = get_arch_name(model_name)
    return _model_entrypoints[arch_name]


def generate_default_cfgs(cfgs: Dict[str, Union[Dict[str, Any], PretrainedCfg]]):
    out = defaultdict(DefaultCfg)
    default_set = set()  # no tag and tags ending with * are prioritized as default

    for k, v in cfgs.items():
        if isinstance(v, dict):
            v = PretrainedCfg(**v)
        has_weights = v.has_weights

        model, tag = split_model_name_tag(k)
        is_default_set = model in default_set
        priority = (has_weights and not tag) or (tag.endswith('*') and not is_default_set)
        tag = tag.strip('*')

        default_cfg = out[model]

        if priority:
            default_cfg.tags.appendleft(tag)
            default_set.add(model)
        elif has_weights and not default_cfg.is_pretrained:
            default_cfg.tags.appendleft(tag)
        else:
            default_cfg.tags.append(tag)

        if has_weights:
            default_cfg.is_pretrained = True

        default_cfg.cfgs[tag] = v

    return out


def register_model(fn: Callable[..., Any]) -> Callable[..., Any]:
    # lookup containing module
    mod = sys.modules[fn.__module__]

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]  # type: ignore

    # add entries to registry dict/sets
    if model_name in _model_entrypoints:
        warnings.warn(
            f'Overwriting {model_name} in registry with {fn.__module__}.{model_name}. This is because the name being '
            'registered conflicts with an existing name. Please check if this is not expected.',
            stacklevel=2,
        )
    _model_entrypoints[model_name] = fn
    # _model_to_module[model_name] = module_name
    # _module_to_models[module_name].add(model_name)
    if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
        # this will catch all models that have entrypoint matching cfg key, but miss any aliasing
        # entrypoints or non-matching combos
        default_cfg = mod.default_cfgs[model_name]
        if not isinstance(default_cfg, DefaultCfg):
            # new style default cfg dataclass w/ multiple entries per model-arch
            assert isinstance(default_cfg, dict)
            # old style cfg dict per model-arch
            pretrained_cfg = PretrainedCfg(**default_cfg)
            default_cfg = DefaultCfg(tags=deque(['']), cfgs={'': pretrained_cfg})

        for tag_idx, tag in enumerate(default_cfg.tags):
            is_default = tag_idx == 0
            pretrained_cfg = default_cfg.cfgs[tag]
            model_name_tag = '.'.join([model_name, tag]) if tag else model_name
            replace_items = dict(architecture=model_name, tag=tag if tag else None)
            if pretrained_cfg.hf_hub_id and pretrained_cfg.hf_hub_id == 'timm/':
                # auto-complete hub name w/ architecture.tag
                replace_items['hf_hub_id'] = pretrained_cfg.hf_hub_id + model_name_tag
            pretrained_cfg = replace(pretrained_cfg, **replace_items)

            if is_default:
                _model_pretrained_cfgs[model_name] = pretrained_cfg
                if pretrained_cfg.has_weights:
                    # add tagless entry if it's default and has weights
                    _model_has_pretrained.add(model_name)

            if tag:
                _model_pretrained_cfgs[model_name_tag] = pretrained_cfg
                if pretrained_cfg.has_weights:
                    # add model w/ tag if tag is valid
                    _model_has_pretrained.add(model_name_tag)
                _model_with_tags[model_name].append(model_name_tag)
            else:
                _model_with_tags[model_name].append(model_name)  # has empty tag (to slowly remove these instances)

        _model_default_cfgs[model_name] = default_cfg

    return fn
