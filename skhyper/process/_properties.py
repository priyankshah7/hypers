import numpy as np

from skhyper.utils import HyperanalysisError


# TODO consider creating a class containing the functions
def data_shape(data):
    """
    :param data:
    :return shape, dimensions:

    Returns the shape and dimensions of the data array
    """
    if type(data) != np.ndarray:
        raise HyperanalysisError('Data must be a numpy array.')

    shape = data.shape
    dimensions = len(shape)

    return shape, dimensions


def data_tranform2d(data):
    """
    :param data:
    :return data2d:

    Returns the data array reshaped into 2-dimensions
    """
    shape, dimensions = data_shape(data)

    if dimensions == 1:
        raise HyperanalysisError('Data must have dimensions between 2 and 4')

    elif dimensions == 2:
        return data

    elif dimensions == 3:
        return np.reshape(data, (shape[0]*shape[1], shape[2]))

    elif dimensions == 4:
        return np.reshape(data, (shape[0]*shape[1]*shape[2], shape[3]))

    else:
        raise HyperanalysisError('Error when reshaping data array into 2-dimensions')


def data_back_transform(data2d, shape, dimensions):
    """
    :param data2d:
    :param shape:
    :param dimensions:
    :return dataTransformed:

    Returns the 2d data array reshaped into specified dimensions
    """
    if dimensions == 3:
        return np.reshape(data2d, (shape[0], shape[1], shape[2]))

    elif dimensions == 4:
        return np.reshape(data2d, (shape[0], shape[1], shape[2], shape[3]))
