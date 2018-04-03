import os
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import InterpolatedUnivariateSpline, interp2d

from skhyper.process import data_shape, data_tranform2d, data_back_transform


def anscombe_transform(data, gauss_std, gauss_mean=0, poisson_multi=1):
    SMALL_VAL = 1
    shape, dimensions = data_shape(data)
    data = data_tranform2d(data)

    data_anscombe = 2 / poisson_multi * np.sqrt(np.fmax(SMALL_VAL, poisson_multi * data +
                                                        (3/8) * poisson_multi ** 2 +
                                                        gauss_std ** 2 -
                                                        poisson_multi * gauss_mean))

    data_anscombe = data_back_transform(data_anscombe, shape, dimensions)

    return data_anscombe


def inverse_anscombe_transform(data, gauss_std, gauss_mean=0, poisson_multi=1):
    shape, dimensions = data_shape(data)
    data = data_tranform2d(data)

    resource_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'anscombe_matrices')
    mat_dict = loadmat(os.path.join(resource_dir, 'general_anscombe_vectors.mat'))

    efzmatrix = np.squeeze(mat_dict['Efzmatrix'])
    ez = np.squeeze(mat_dict['Ez'])
    sigmas = np.squeeze(mat_dict['sigmas'])

    gauss_std = gauss_std / poisson_multi

    if gauss_std > np.max(sigmas):
        exact_inverse = anscombe_inverse_exact_unbiased(data) - gauss_std ** 2
        exact_inverse = np.fmax(np.zeros(exact_inverse.shape), exact_inverse)

    elif gauss_std > 0:
        efz = interp2d(sigmas, ez, efzmatrix, kind='linear')(gauss_std, ez)
        exact_inverse = InterpolatedUnivariateSpline(efz, ez, k=1)(data)

        asymptotic = anscombe_inverse_exact_unbiased(data) - gauss_std ** 2

        outside_exact_inverse_domain = data > np.max(efz.flatten())
        exact_inverse[outside_exact_inverse_domain] = asymptotic[outside_exact_inverse_domain]

        outside_exact_inverse_domain = data < np.min(efz)
        exact_inverse[outside_exact_inverse_domain] = 0

    elif gauss_std == 0:
        exact_inverse = anscombe_inverse_exact_unbiased(data)

    else:
        raise ValueError('Error: gaussStd must be non-negative.')

    exact_inverse *= poisson_multi
    exact_inverse += gauss_mean

    exact_inverse = data_back_transform(exact_inverse, shape, dimensions)

    return exact_inverse


def anscombe_inverse_exact_unbiased(data):
    resource_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'anscombe_matrices')
    mat_dict = loadmat(os.path.join(resource_dir, 'anscombe_vectors.mat'))

    Efz = mat_dict['Efz']
    Ez = mat_dict['Ez']

    asymptotic = (data / 2)**2 - 1/8
    signal = InterpolatedUnivariateSpline(Efz, Ez, k=1)(data)

    outside_exact_inverse_domain = data > np.max(Efz)
    signal[outside_exact_inverse_domain] = asymptotic[outside_exact_inverse_domain]

    outside_exact_inverse_domain = data < 2 * np.sqrt(3/8)
    signal[outside_exact_inverse_domain] = 0

    return signal
