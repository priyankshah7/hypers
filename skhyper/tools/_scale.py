import numpy as np 


def _data_scale(X):
    """ Scale the hyperspectral data

    Scales the hyperspectral data to between 0 and 1 for all positive data or
    -1 and 1 for positive and negative data.
    """
    X.data = X.data / np.max(np.abs(X.data))
