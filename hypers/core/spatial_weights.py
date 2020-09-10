import numpy as np
from pysal.lib.weights import KNN, W

__all__ = ['spatial_weights_matrix']


def spatial_weights_matrix(image_shape: tuple, k: int = 4) -> W:
    """
    Generate a spatial weights matrix based on k-nearest neighbours

    Parameters
    ----------
    image_shape: tuple
        A tuple of 2 integers representing the size of the original image.
    k: int
        Number of nearest neighbours to consider

    Returns
    -------
    pysal.lib.weights.W
        An object containing information about the spatial weights matrix generated.
        A sparse matrix of the spatial weights matrix can be accessed by using the
        sparse attribute W.sparse
    """
    x, y = np.indices(image_shape)
    x.shape = (np.prod(image_shape), 1)
    y.shape = (np.prod(image_shape), 1)
    arr = np.hstack([x, y])
    w = KNN(arr, k=k)
    return w
