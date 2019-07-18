import numpy as np
from typing import Union
from hypers.core import hparray


def array(input_array: Union[list, np.ndarray, hparray]) -> hparray:
    """
    Returns a hparray object.

    Parameters
    ----------
    input_array: list, np.ndarray or hp.hparray
        The input hyperspectral array

    Returns
    -------
    hp.hparray
        The hypers array object
    """
    return hparray(input_array)
