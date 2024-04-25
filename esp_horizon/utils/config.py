from collections import OrderedDict


def as_flat_config(config, separator=".", flatten_lists=True):
    """Convert nested config to flat config."""
    if isinstance(config, (tuple, list)):
        if not flatten_lists:
            return config
        config = OrderedDict([(str(i), v) for i, v in enumerate(config)])
    if not isinstance(config, (dict, OrderedDict)):
        raise TypeError("Expected dictionary, got {}.".format(type(config)))
    config_classes = (dict, OrderedDict) + ((tuple, list) if flatten_lists else tuple())
    flat = OrderedDict()
    for k, v in config.items():
        if isinstance(v, config_classes):
            for sk, sv in as_flat_config(v, separator=separator, flatten_lists=flatten_lists).items():
                flat[k + separator + sk] = sv
        else:
            flat[k] = v
    return flat
