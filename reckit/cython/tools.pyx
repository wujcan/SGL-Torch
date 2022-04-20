# distutils: language = c++
__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

import numpy as np

__all__ = ["float_type", "int_type", "is_ndarray", "as_np_array"]

def get_float_type():
    cdef size_of_float = sizeof(float)*8
    if size_of_float == 32:
        return np.float32
    elif size_of_float == 64:
        return np.float64
    else:
        raise EnvironmentError("The size of 'float' is %d, but 32 or 64." % size_of_float)

def get_int_type():
    cdef size_of_int = sizeof(int)*8
    if size_of_int == 16:
        return np.int16
    elif size_of_int == 32:
        return np.int32
    else:
        raise EnvironmentError("The size of 'int' is %d, but 16 or 32." % size_of_int)


float_type = get_float_type()
int_type = get_int_type()


def is_ndarray(array, dtype):
    if not isinstance(array, np.ndarray):
        return False
    if array.dtype != dtype:
        return False
    if array.base is not None:
        return False
    return True


def get_elem_type(data):
    type_map = {np.float32: float_type, np.float64: float_type,
                np.int32: int_type, np.int64: int_type}
    if isinstance(data, np.ndarray):
        if data.dtype == np.float32 or data.dtype == np.float64:
            dtype = float_type
        if data.dtype == np.int32 or data.dtype == np.int64:
            dtype = int_type
    elif isinstance(data, (list, tuple)):
        dtype = get_elem_type(data[0])
    elif isinstance(data, float):
        dtype = float_type
    elif isinstance(data, int):
        dtype = int_type
    else:
        dtype = None

    return dtype


def as_np_array(array, dtype=None, copy=False):
    if dtype is None:
        dtype = get_elem_type(array)

    if dtype is None:
        raise TypeError(f"{type(array), array.dtype} is not supported.")

    if not is_ndarray(array, dtype):
        copy = True

    array = np.array(array, dtype=dtype, copy=copy)

    return array
