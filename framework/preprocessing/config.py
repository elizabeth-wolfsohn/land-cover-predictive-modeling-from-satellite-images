"""
Helpers to read optional values from the YAML config (a YamlNamespace, a dict, or None).
"""


def cfg_get(cfg, path, default=None):
    """Read a possibly nested, possibly missing config value.

    Args:
        cfg (YamlNamespace | dict | None): the config (or a sub-section of it)
        path (str): dot-separated key path, e.g. 'tiling.resampling.threshold'
        default: value returned when any key along the path is missing or null
    Returns:
        the value, or `default`
    """
    cur = cfg
    for key in path.split('.'):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def section_enabled(cfg, name):
    """True if the config has a section `name` with `enabled: true`."""
    return bool(cfg_get(cfg, f'{name}.enabled', False))
