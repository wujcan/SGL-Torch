# distutils: language = c++
__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

from libcpp.unordered_set cimport unordered_set as cset
from libcpp.vector cimport vector as cvector
import numpy as np
cimport numpy as np
from .tools import float_type, int_type, is_ndarray

ctypedef cset[int] int_set

cdef extern from "include/randint.h":
    int c_randint_choice(int high, int size, int replace, const float* prob, const int_set* exclusion, int* result)
    int c_batch_randint_choice(int high, const int* size_ptr, int batch_num, int replace,
                               const float* prob, const int_set* exclusion, int n_threads, int* result)


def pyx_randint_choice(high, size=1, replace=True, p=None, exclusion=None):
    """Sample random integers from [0, high).

    Args:
        high (int): The largest integer (exclusive) to be drawn from the distribution.
        size (int): The number of samples to be drawn.
        replace (bool): Whether the sample is with or without replacement.
        p: 1-D array-like, optional. The probabilities associated with each entry in [0, high).
           If not given the sample assumes a uniform distribution.
        exclusion: 1-D array-like. The integers in exclusion will be excluded.

    Returns:

    """
    if high <= 1:
        raise ValueError("'high' must be larger than 1.")
    if size <= 0:
        raise ValueError("'size' must be a positive integer.")

    if not isinstance(replace, bool):
        raise TypeError("'replace' must be bool.")

    if p is not None:
        if not is_ndarray(p, float_type):
            p = np.array(p, dtype=float_type)
        if p.ndim != 1:
            raise ValueError("'p' must be a 1-dim array_like")
        if len(p) != high:
            raise ValueError("The length of 'p' must be equal with 'high'.")

    if exclusion is not None and len(exclusion) >= high:
        raise ValueError("The length of 'exclusion' must be smaller than 'high'.")

    len_exclusion = len(exclusion) if exclusion is not None else 0
    if replace is False and (high-len_exclusion <= size):
        raise ValueError("There is not enough integers to be sampled.")

    if isinstance(exclusion, (int, int_type)):
        exclusion = [exclusion]

    cdef int_set* exc_ptr = <int_set*> 0
    cdef int_set _exclusion
    if exclusion is not None:
        _exclusion = exclusion
        exc_ptr = &_exclusion

    cdef float* p_pt = <float*> 0
    if p is not None:
        p_pt = <float *>np.PyArray_DATA(p)

    results = np.zeros(size, dtype=int_type)
    results_pt = <int *>np.PyArray_DATA(results)

    c_randint_choice(high, size, replace, p_pt, exc_ptr, results_pt)

    if len(results) == 1:
        results = results[0]
    return results


def pyx_batch_randint_choice(high, size, replace=True, p=None, exclusion=None, thread_num=1):
    """Sample random integers from [0, high).

    Args:
        high (int):
        size: 1-D array_like
        replace (bool):
        p: 2-D array_like
        exclusion: a list of 1-D array_like

    Returns:
        list: a list of 1-D array_like sample
    """
    if high <= 1:
        raise ValueError("'high' must be larger than 1.")

    if not isinstance(replace, bool):
        raise TypeError("'replace' must be bool.")

    if thread_num < 1 or not isinstance(thread_num, (int, int_type)):
        raise ValueError("'thread_num' must be a positive integer.")

    try:
        size = np.array(size, int_type)
    except:
        raise ValueError("'size' must be a 1-dim array_like of positive integers.")
    if size.ndim != 1 or np.any(size<=0):
        raise ValueError("'size' must be a 1-dim array_like of positive integers.")

    cdef int* size_ptr = <int *>np.PyArray_DATA(size)

    cdef float* p_ptr = <float*> 0
    if p is not None:
        if not is_ndarray(p, float_type):
            p = np.array(p, dtype=float_type)
        if p.ndim != 2:
            raise ValueError("'p' must be a 2-dim array_like.")

        nrow, ncol = np.shape(p)
        if nrow != len(size):
            raise ValueError("The number of rows of 'p' must be equal with the length of 'size'.")
        if ncol != high:
            raise ValueError("The number of columns of 'p' must be equal with 'high'.")

        p_ptr = <float*> np.PyArray_DATA(p)

    cdef int_set* exc_ptr = <int_set*> 0
    cdef cvector[int_set] c_exclusion
    if exclusion is not None:
        if len(exclusion) != len(size):
            raise ValueError("The length of 'exclusion' must be equal with the length of 'size'.")
        for idx, (exc, s) in enumerate(zip(exclusion, size)):
            if len(exc) >= high:
                raise ValueError("The length of 'exclusion' must be smaller than 'high' in %d-th row." % idx)

            len_exc = len(exc)
            if replace is False and (high-len_exc <= s):
                raise ValueError("There is not enough integers to be sampled in %d-th row." % idx)

        for exc in exclusion:
            c_exclusion.push_back(exc)
        exc_ptr = &c_exclusion[0]

    cdef int batch_num = len(size)
    results = np.zeros(np.sum(size), dtype=int_type)
    cdef int* result_ptr = <int*> np.PyArray_DATA(results)

    c_batch_randint_choice(high, size_ptr, batch_num, replace,
                           p_ptr, exc_ptr, thread_num, result_ptr)

    indices = np.cumsum(size)[0:-1]
    return np.split(results, indices)
