import numpy as np

from skhyper.process import data_shape


def _data_checks(data):
    shape, dimensions = data_shape(data)

    if type(data) != np.ndarray:
        raise TypeError('Data must be a numpy array.')

    if dimensions != 3 and dimensions != 4:
        raise TypeError('Data must be 3- or 4- dimensional.')

    return shape, dimensions
