import numpy as np
from typing import Union
from hypers.core.array import hparray

# types that can be used for arrays
ListOrArray = Union[list, np.ndarray, hparray]
MixedArray = Union[np.ndarray, hparray]


def convert_nparray(array: ListOrArray):
    """
    Convert a list or hypers array to a numpy array.

    Parameters
    ----------
    array: ListOrArray
        Variable of type list, `np.ndarray` or `hp.hparray`.

    Returns
    -------
    np.ndarray
        Converted array.
    """
    return np.asarray(array)
