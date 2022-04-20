__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["randint_choice", "batch_randint_choice"]

from reckit.cython import pyx_randint_choice, pyx_batch_randint_choice


def randint_choice(high, size=1, replace=True, p=None, exclusion=None):
    """Sample random integers from [0, high).

    Args:
        high (int): The largest integer (exclusive) to be drawn from the distribution.
        size (int): The number of samples to be drawn.
        replace (bool): Whether the sample is with or without replacement.
        p: 1-D array-like, optional. The probabilities associated with each entry in [0, high).
           If not given the sample assumes a uniform distribution.
        exclusion: 1-D array-like. The integers in exclusion will be excluded.

    Returns:
        int or ndarray
    """
    return list(pyx_randint_choice(high, size, replace, p, exclusion))


def batch_randint_choice(high, size, replace=True, p=None, exclusion=None, thread_num=1):
    """Sample random integers from [0, high).

    Args:
        high (int):
        size: 1-D array_like
        replace (bool):
        p: 2-D array_like
        exclusion: a list of 1-D array_like
        thread_num (int): the number of threads

    Returns:
        list: a list of 1-D array_like sample
    """
    return pyx_batch_randint_choice(high, size, replace=replace, p=p, exclusion=exclusion, thread_num=thread_num)
