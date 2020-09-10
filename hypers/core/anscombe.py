import os
import numpy as np
from typing import Union
from scipy.io import loadmat
from scipy.interpolate import InterpolatedUnivariateSpline, interp2d

import hypers as hp

__all__ = ['anscombe_transformation', 'inverse_anscombe_transformation']

resource_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)))


def anscombe_transformation(signal: Union[list, np.ndarray, 'hp.hparray'],
                            gauss_std: float, gauss_mean: float=0, poisson_multi: float=1):
    """
    Anscombe transformation

    Parameters
    ----------
    signal: Union[list, np.ndarray, hp.hparray]
        The array to perform the transformation on.

    gauss_std: float
        The standard deviation of the noise.

    gauss_mean: float
        The mean value of the noise.

    poisson_multi: float
        Poisson multiplier.

    Returns
    -------
    Union[np.ndarray, hp.hparray]
        The transformed signal.
    """
    SMALL_VAL = 0
    if isinstance(signal, list):
        signal = np.array(signal)

    fsignal = 2 / poisson_multi * np.sqrt(np.fmax(SMALL_VAL, poisson_multi * signal +
                                                  (3 / 8) * poisson_multi ** 2 +
                                                  gauss_std ** 2 -
                                                  poisson_multi * gauss_mean))
    return fsignal


def inverse_anscombe_transformation(fsignal: Union[list, np.ndarray, 'hp.hparray'],
                                    gauss_std: float, gauss_mean: float=0, poisson_multi: float=1):
    """
    Inverse Anscombe transformation

    Parameters
    ----------
    fsignal: Union[list, np.ndarray, hp.hparray]
        The array to perform the transformation on.

    gauss_std: float
        The standard deviation of the noise.

    gauss_mean: float
        The mean value of the noise.

    poisson_multi: float
        Poisson multiplier.

    Returns
    -------
    Union[np.ndarray, hp.hparray]
        The transformed signal.
    """
    if isinstance(fsignal, list):
        fsignal = np.array(fsignal)

    mat_dict = loadmat(os.path.join(resource_dir, 'gen_anscombe_vectors.mat'))
    Efzmatrix = np.squeeze(mat_dict['Efzmatrix'])
    Ez = np.squeeze(mat_dict['Ez'])
    sigmas = np.squeeze(mat_dict['sigmas'])
    gauss_std = gauss_std / poisson_multi

    # interpolate the exact unbiased inverse for the desired gauss_std
    # gauss_std is given as input parameter
    if gauss_std > np.max(sigmas):
        # for very large sigmas, use the exact unbiased inverse of
        # Anscombe modified by a -gauss_std^2 addend
        exact_inverse = _anscombe_inverse_exact_unbiased(fsignal) - gauss_std ** 2
        # this should be necessary, since anscombe_inverse_exact_unbiased(fsignal) is >=0 and gauss_std>=0.
        exact_inverse = np.fmax(np.zeros(exact_inverse.shape), exact_inverse)

    elif gauss_std > 0:
        # interpolate Efz
        Efz = interp2d(sigmas, Ez, Efzmatrix, kind='linear')(gauss_std, Ez)

        # apply the exact unbiased inverse
        exact_inverse = InterpolatedUnivariateSpline(Efz, Ez, k=1)(fsignal)

        # outside the pre-computed domain, use the exact unbiased inverse
        # of Anscombe modified by a -gauss_std^2 addend
        # (the exact unbiased inverse of Anscombe takes care of asymptotics)
        outside_exact_inverse_domain = fsignal > np.max(Efz.flatten())
        asymptotic = _anscombe_inverse_exact_unbiased(fsignal) - gauss_std ** 2
        exact_inverse[outside_exact_inverse_domain] = asymptotic[outside_exact_inverse_domain]
        outside_exact_inverse_domain = fsignal < np.min(Efz)
        exact_inverse[outside_exact_inverse_domain] = 0

    elif gauss_std == 0:
        # if gauss_std is zero, then use exact unbiased inverse of Anscombe
        # transformation (higher numerical precision)
        exact_inverse = _anscombe_inverse_exact_unbiased(fsignal)

    else:
        raise ValueError('gauss_std must be non-negative.')

    exact_inverse *= poisson_multi
    exact_inverse += gauss_mean

    return exact_inverse


def _anscombe_inverse_exact_unbiased(fsignal: Union[list, np.ndarray, 'hp.hparray']):
    """
    Calculate exact inverse Anscombe transformation

    Parameters
    ----------
    fsignal : Union[list, np.ndarray, hp.hparray]
        Forward Anscombe-transformed noisy signal

    Returns
    -------
    signal : Union[np.ndarray, hp.hparray]
        Inverse Anscombe-transformed signal
    """
    if isinstance(fsignal, list):
        fsignal = np.array(fsignal)

    mat_dict = loadmat(os.path.join(resource_dir, 'anscombe_vectors.mat'))
    Efz = mat_dict['Efz']
    Ez = mat_dict['Ez']
    asymptotic = (fsignal / 2) ** 2 - 1 / 8  # asymptotically unbiased inverse [3]
    signal = InterpolatedUnivariateSpline(Efz, Ez, k=1)(fsignal)  # exact unbiased inverse [1,2]
    # for large values use asymptotically unbiased inverse instead
    # of linear extrapolation of exact unbiased inverse outside of pre-computed domain
    outside_exact_inverse_domain = fsignal > np.max(Efz)
    signal[outside_exact_inverse_domain] = asymptotic[outside_exact_inverse_domain]
    outside_exact_inverse_domain = fsignal < 2 * np.sqrt(3 / 8)  # min(Efz(:));
    signal[outside_exact_inverse_domain] = 0

    return signal
