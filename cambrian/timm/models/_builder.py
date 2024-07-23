import logging
import dataclasses
from typing import Optional, Dict, Callable, Any, Tuple

from cambrian.timm.models._pretrained import PretrainedCfg

_logger = logging.getLogger(__name__)


def _filter_kwargs(kwargs, names):
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)


def _update_default_model_kwargs(pretrained_cfg, kwargs, kwargs_filter):
    """ Update the default_cfg and kwargs before passing to model

    Args:
        pretrained_cfg: input pretrained cfg (updated in-place)
        kwargs: keyword args passed to model build fn (updated in-place)
        kwargs_filter: keyword arg keys that must be removed before model __init__
    """
    # Set model __init__ args that can be determined by default_cfg (if not already passed as kwargs)
    default_kwarg_names = ('num_classes', 'global_pool', 'in_chans')
    if pretrained_cfg.get('fixed_input_size', False):
        # if fixed_input_size exists and is True, model takes an img_size arg that fixes its input size
        default_kwarg_names += ('img_size',)

    for n in default_kwarg_names:
        # for legacy reasons, model __init__args uses img_size + in_chans as separate args while
        # pretrained_cfg has one input_size=(C, H ,W) entry
        if n == 'img_size':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        elif n == 'num_classes':
            default_val = pretrained_cfg.get(n, None)
            # if default is < 0, don't pass through to model
            if default_val is not None and default_val >= 0:
                kwargs.setdefault(n, pretrained_cfg[n])
        else:
            default_val = pretrained_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, pretrained_cfg[n])

    # Filter keyword args for task specific model variants (some 'features only' models, etc.)
    _filter_kwargs(kwargs, names=kwargs_filter)


def resolve_pretrained_cfg(
        variant: str,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
) -> PretrainedCfg:
    model_with_tag = variant
    pretrained_tag = None
    if pretrained_cfg:
        if isinstance(pretrained_cfg, dict):
            # pretrained_cfg dict passed as arg, validate by converting to PretrainedCfg
            pretrained_cfg = PretrainedCfg(**pretrained_cfg)
        elif isinstance(pretrained_cfg, str):
            pretrained_tag = pretrained_cfg
            pretrained_cfg = None

    # fallback to looking up pretrained cfg in model registry by variant identifier
    if not pretrained_cfg:
        # if pretrained_tag:
        #     model_with_tag = '.'.join([variant, pretrained_tag])
        # pretrained_cfg = get_pretrained_cfg(model_with_tag)
        raise NotImplementedError

    if not pretrained_cfg:
        _logger.warning(
            f"No pretrained configuration specified for {model_with_tag} model. Using a default."
            f" Please add a config to the model pretrained_cfg registry or pass explicitly.")
        pretrained_cfg = PretrainedCfg()  # instance with defaults

    pretrained_cfg_overlay = pretrained_cfg_overlay or {}
    if not pretrained_cfg.architecture:
        pretrained_cfg_overlay.setdefault('architecture', variant)
    pretrained_cfg = dataclasses.replace(pretrained_cfg, **pretrained_cfg_overlay)

    return pretrained_cfg


def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        pretrained_cfg: Optional[Dict] = None,
        pretrained_cfg_overlay: Optional[Dict] = None,
        model_cfg: Optional[Any] = None,
        feature_cfg: Optional[Dict] = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Optional[Callable] = None,
        kwargs_filter: Optional[Tuple[str]] = None,
        **kwargs,
):
    """ Build model with specified default_cfg and optional model_cfg

    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretrained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation
      * pruning config / model adaptation

    Args:
        model_cls (nn.Module): model class
        variant (str): model variant name
        pretrained (bool): load pretrained weights
        pretrained_cfg (dict): model's pretrained weight/task config
        model_cfg (Optional[Dict]): model's architecture config
        feature_cfg (Optional[Dict]: feature extraction adapter config
        pretrained_strict (bool): load pretrained weights strictly
        pretrained_filter_fn (Optional[Callable]): filter callable for pretrained weights
        kwargs_filter (Optional[Tuple]): kwargs to filter before passing to model
        **kwargs: model args passed through to model __init__
    """
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}

    # resolve and update model pretrained config and model kwargs
    pretrained_cfg = resolve_pretrained_cfg(
        variant,
        pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay=pretrained_cfg_overlay
    )

    # FIXME converting back to dict, PretrainedCfg use should be propagated further, but not into model
    pretrained_cfg = pretrained_cfg.to_dict()

    _update_default_model_kwargs(pretrained_cfg, kwargs, kwargs_filter)

    # Setup for feature extraction wrapper done at end of this fn
    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')

    # Instantiate the model
    if model_cfg is None:
        model = model_cls(**kwargs)
    else:
        model = model_cls(cfg=model_cfg, **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat

    if pruned:
        # model = adapt_model_from_file(model, variant)
        raise NotImplementedError

    # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        # load_pretrained(
        #     model,
        #     pretrained_cfg=pretrained_cfg,
        #     num_classes=num_classes_pretrained,
        #     in_chans=kwargs.get('in_chans', 3),
        #     filter_fn=pretrained_filter_fn,
        #     strict=pretrained_strict,
        # )
        raise NotImplementedError

    # Wrap the model in a feature extraction module if enabled
    if features:
        # feature_cls = FeatureListNet
        # output_fmt = getattr(model, 'output_fmt', None)
        # if output_fmt is not None:
        #     feature_cfg.setdefault('output_fmt', output_fmt)
        # if 'feature_cls' in feature_cfg:
        #     feature_cls = feature_cfg.pop('feature_cls')
        #     if isinstance(feature_cls, str):
        #         feature_cls = feature_cls.lower()
        #         if 'hook' in feature_cls:
        #             feature_cls = FeatureHookNet
        #         elif feature_cls == 'fx':
        #             feature_cls = FeatureGraphNet
        #         else:
        #             assert False, f'Unknown feature class {feature_cls}'
        # model = feature_cls(model, **feature_cfg)
        # model.pretrained_cfg = pretrained_cfg_for_features(pretrained_cfg)  # add back pretrained cfg
        # model.default_cfg = model.pretrained_cfg  # alias for rename backwards compat (default_cfg -> pretrained_cfg)
        raise NotImplementedError

    return model
