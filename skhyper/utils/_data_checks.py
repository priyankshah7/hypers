import numpy as np

from skhyper.utils import HyperanalysisError
from skhyper.process import data_shape


def _data_checks(data):
    shape, dimensions = data_shape(data)

    if type(data) != np.ndarray:
        raise HyperanalysisError('Data must be a numpy array.')

    if dimensions != 3 and dimensions != 4:
        raise HyperanalysisError('Data must be 3- or 4- dimensional.')

    return shape, dimensions
