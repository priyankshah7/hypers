import numpy as np
import scipy.optimize as opt

import hypers as hp

__all__ = ['ucls', 'nnls']


def ucls(X: 'hp.hparray', x_fit: np.ndarray) -> np.ndarray:
    """
    Unconstrained least-squares.

    Parameters
    ----------
    X: hp.hparray
        hypers hyperspectral array.
    x_fit: np.ndarray
        Array of spectra to fit to the hyperspectral data. Must be of
        size (nspectral, n_fit) where n_fit is the number of spectra
        provided to fit.

    Returns
    -------
    np.ndarray
        Array of images with an image per spectrum to fit. Array has
        size of (nspatial, n_fit).
    """
    x_inverse = np.linalg.pinv(x_fit)
    maps = np.dot(x_inverse, X.collapse().T).T.reshape(X.shape[:-1] + (x_fit.shape[-1],))

    return maps


def nnls(X: 'hp.hparray', x_fit: np.ndarray) -> np.ndarray:
    """
    Non-negative least-squares.

    Parameters
    ----------
    X: hp.hparray
        hypers hyperspectral array.
    x_fit: np.ndarray
        Array of spectra to fit to the hyperspectral data. Must be of
        size (nspectral, n_fit) where n_fit is the number of spectra
        provided to fit.

    Returns
    -------
    np.ndarray
        Array of images with an image per spectrum to fit. Array has
        size of (nspatial, n_fit).
    """
    M = X.collapse()

    N, p1 = M.shape
    q, p2 = x_fit.T.shape

    maps = np.zeros((N, q), dtype=np.float32)
    MtM = np.dot(x_fit.T, x_fit)
    for n1 in range(N):
        maps[n1] = opt.nnls(MtM, np.dot(x_fit.T, M[n1]))[0]
    maps = maps.reshape(X.shape[:-1] + (x_fit.shape[-1],))

    return maps
