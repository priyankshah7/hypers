"""
Utilities to perform checks on hyperspectral data
"""
import numpy as np

from skhyper.process import data_shape


def _data_checks(X):
    shape, dimensions = data_shape(X)
    if type(X) != np.ndarray:
        raise TypeError('Data must be a numpy array.')

    if dimensions != 3 and dimensions != 4:
        raise TypeError('Data must be 3- or 4- dimensional.')

    return shape, dimensions


def _check_features_samples(X):
    """
    Performs a check to ensure that the number of samples are greater than the number of features.
    """
    _shape, _dimensions = data_shape(X)
    if _dimensions == 3:
        if not (_shape[0] * _shape[1]) > _shape[2]:
            raise TypeError('The number of samples must be greater than the number of features')

    elif _dimensions == 4:
        if not (_shape[0] * _shape[1] * _shape[2]) > _shape[3]:
            raise TypeError('The number of samples must be greater than the number of features')
