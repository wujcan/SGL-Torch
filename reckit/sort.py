__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["sort", "arg_sort", "top_k", "arg_top_k"]

from reckit.cython.sort import pyx_sort, pyx_arg_sort, pyx_top_k, pyx_arg_top_k


def sort(array, reverse=False, n_threads=1):
    """Return a sorted copy of an array along the last axis

    Args:
        array: 1-dim or 2-dim array_like.
        reverse (bool): reverse flag can be set to request the result in descending order.
        n_threads (int):

    Returns:

    """
    return pyx_sort(array, reverse, n_threads)


def arg_sort(array, reverse=False, n_threads=1):
    """Returns the indices that would sort an array along the last axis.

    Args:
        array: 1-dim or 2-dim array_like.
        reverse(bool): reverse flag can be set to request the result in descending order.
        n_threads(int):

    Returns:

    """
    return pyx_arg_sort(array, reverse, n_threads)


def top_k(array, topk, n_threads=1):
    """Return top-k elements along the last axis

    Args:
        array: 1-dim or 2-dim array_like.
        topk(int):
        n_threads(int):

    Returns:

    """
    return pyx_top_k(array, topk, n_threads)


def arg_top_k(array, topk, n_threads=1):
    """Return the indices of top-k elements along the last axis.

    Args:
        array: 1-dim or 2-dim array_like.
        topk(int):
        n_threads(int):

    Returns:

    """
    return pyx_arg_top_k(array, topk, n_threads)
