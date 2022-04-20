# distutils: language = c++
__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

import numpy as np
cimport numpy as np
from .tools import float_type, int_type, as_np_array


cdef extern from "include/sort.h":
    int c_sort_1d[T](const T *array_ptr, int length, T *result, int reverse)
    int c_sort_2d[T](const T *matrix_ptr, int n_rows, int n_cols, int n_threads, T *results_ptr, int reverse)

    int c_arg_sort_1d[T](const T *array_ptr, int length, int *result, int reverse)
    int c_arg_sort_2d[T](const T *matrix_ptr, int n_rows, int n_cols, int n_threads, int *results_ptr, int reverse)

    int c_top_k_1d[T](const T *array_ptr, int length, int top_k, T *result)
    int c_top_k_2d[T](const T *matrix_ptr, int n_rows, int n_cols, int top_k, int n_threads, T *results_ptr)

    int c_arg_top_k_1d[T](const T *array_ptr, int length, int top_k, int *result)
    int c_arg_top_k_2d[T](const T *matrix_ptr, int n_rows, int n_cols, int top_k, int n_threads, int *results_ptr)


def pyx_sort(array, reverse=False, n_threads=1):
    """Return a sorted copy of an array along the last axis

    Args:
        array: 1-dim or 2-dim array_like.
        reverse (bool): reverse flag can be set to request the result in descending order.
        n_threads (int):

    Returns:

    """
    array = as_np_array(array)
    array_ptr = np.PyArray_DATA(array)

    if array.ndim > 2:
        raise ValueError("'array' must be 1-dim or 2-dim array_like.")

    result = np.zeros_like(array)
    result_ptr = np.PyArray_DATA(result)

    if array.dtype == int_type:
        if array.ndim == 1:
            n_dims = array.shape[0]
            c_sort_1d(<int*>array_ptr, n_dims, <int*>result_ptr, reverse)
        if array.ndim == 2:
            n_rows, n_cols = array.shape
            c_sort_2d(<int*>array_ptr, n_rows, n_cols, n_threads, <int*>result_ptr, reverse)
    elif array.dtype == float_type:
        if array.ndim == 1:
            n_dims = array.shape[0]
            c_sort_1d(<float*>array_ptr, n_dims, <float*>result_ptr, reverse)
        if array.ndim == 2:
            n_rows, n_cols = array.shape
            c_sort_2d(<float*>array_ptr, n_rows, n_cols, n_threads, <float*>result_ptr, reverse)
    else:
        raise TypeError(f"'{array.dtype}' is not supported.")

    return result

def pyx_arg_sort(array, reverse=False, n_threads=1):
    """Returns the indices that would sort an array along the last axis.

    Args:
        array: 1-dim or 2-dim array_like.
        reverse(bool): reverse flag can be set to request the result in descending order.
        n_threads(int):

    Returns:

    """
    array = as_np_array(array)
    array_ptr = np.PyArray_DATA(array)

    if array.ndim > 2:
        raise ValueError("'array' must be 1-dim or 2-dim array_like.")

    result = np.zeros_like(array, dtype=int_type)
    result_ptr = np.PyArray_DATA(result)

    if array.dtype == int_type:
        if array.ndim == 1:
            n_dims = array.shape[0]
            c_arg_sort_1d(<int*>array_ptr, n_dims, <int*>result_ptr, reverse)
        if array.ndim == 2:
            n_rows, n_cols = array.shape
            c_arg_sort_2d(<int*>array_ptr, n_rows, n_cols, n_threads, <int*>result_ptr, reverse)
    elif array.dtype == float_type:
        if array.ndim == 1:
            n_dims = array.shape[0]
            c_arg_sort_1d(<float*>array_ptr, n_dims, <int*>result_ptr, reverse)
        if array.ndim == 2:
            n_rows, n_cols = array.shape
            c_arg_sort_2d(<float*>array_ptr, n_rows, n_cols, n_threads, <int*>result_ptr, reverse)
    else:
        raise TypeError("The type of 'array' is not supported.")

    return result


def pyx_top_k(array, top_k, n_threads=1):
    """Return top-k elements along the last axis

    Args:
        array: 1-dim or 2-dim array_like.
        top_k(int):
        n_threads(int):

    Returns:

    """
    array = as_np_array(array)
    array_ptr = np.PyArray_DATA(array)

    if array.ndim == 1:
        result = np.zeros([top_k], dtype=array.dtype)
        result_ptr = np.PyArray_DATA(result)
    elif array.ndim == 2:
        result = np.zeros([array.shape[0], top_k], dtype=array.dtype)
        result_ptr = np.PyArray_DATA(result)
    else:
        raise ValueError("'array' must be 1-dim or 2-dim array_like.")

    if array.dtype == int_type:
        if array.ndim == 1:
            n_dims = array.shape[0]
            c_top_k_1d(<int*>array_ptr, n_dims, top_k, <int*>result_ptr)
        if array.ndim == 2:
            n_rows, n_cols = array.shape
            c_top_k_2d(<int*>array_ptr, n_rows, n_cols, top_k, n_threads, <int*>result_ptr)
    elif array.dtype == float_type:
        if array.ndim == 1:
            n_dims = array.shape[0]
            c_top_k_1d(<float*>array_ptr, n_dims, top_k, <float*>result_ptr)
        if array.ndim == 2:
            n_rows, n_cols = array.shape
            c_top_k_2d(<float*>array_ptr, n_rows, n_cols, top_k, n_threads, <float*>result_ptr)
    else:
        raise TypeError("The type of 'array' is not supported.")

    return result


def pyx_arg_top_k(array, top_k, n_threads=1):
    """Return the indices of top-k elements along the last axis.

    Args:
        array: 1-dim or 2-dim array_like.
        top_k(int):
        n_threads(int):

    Returns:

    """
    array = as_np_array(array)
    array_ptr = np.PyArray_DATA(array)

    if array.ndim == 1:
        result = np.zeros([top_k], dtype=int_type)
        result_ptr = np.PyArray_DATA(result)
    elif array.ndim == 2:
        result = np.zeros([array.shape[0], top_k], dtype=int_type)
        result_ptr = np.PyArray_DATA(result)
    else:
        raise ValueError("'array' must be 1-dim or 2-dim array_like.")

    if array.dtype == int_type:
        if array.ndim == 1:
            n_dims = array.shape[0]
            c_arg_top_k_1d(<int*>array_ptr, n_dims, top_k, <int*>result_ptr)
        if array.ndim == 2:
            n_rows, n_cols = array.shape
            c_arg_top_k_2d(<int*>array_ptr, n_rows, n_cols, top_k, n_threads, <int*>result_ptr)
    elif array.dtype == float_type:
        if array.ndim == 1:
            n_dims = array.shape[0]
            c_arg_top_k_1d(<float*>array_ptr, n_dims, top_k, <int*>result_ptr)
        if array.ndim == 2:
            n_rows, n_cols = array.shape
            c_arg_top_k_2d(<float*>array_ptr, n_rows, n_cols, top_k, n_threads, <int*>result_ptr)
    else:
        raise TypeError("The type of 'array' is not supported.")

    return result
