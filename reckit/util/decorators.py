__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["typeassert", "timer"]

import time
from inspect import signature
from functools import wraps
from collections import Iterable


def typeassert(*type_args, **type_kwargs):
    def decorate(func):
        sig = signature(func)
        bound_types = sig.bind_partial(*type_args, **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    types = bound_types[name]
                    if not isinstance(types, Iterable):
                        types = [types]

                    if value is None:
                        if None in types:
                            continue
                        else:
                            raise TypeError('Argument {} must be {}'.format(name, bound_types[name]))

                    types = tuple([t for t in types if t is not None])
                    if not isinstance(value, types):
                        raise TypeError('Argument {} must be {}'.format(name, bound_types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate


def timer(func):
    """The timer decorator
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("%s function cost: %fs" % (func.__name__, end_time - start_time))
        return result
    return wrapper
