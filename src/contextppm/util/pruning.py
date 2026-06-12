from importlib import import_module


def standard_prune_func(x):
    return 1 - 1 / (x + 1 / 9)


def no_prune_func(x):
    return 0


def load_func(dotpath: str):
    module_, func = dotpath.rsplit(".", maxsplit=1)
    m = import_module(module_)
    return getattr(m, func)
