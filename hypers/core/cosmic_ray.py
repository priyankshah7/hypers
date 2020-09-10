import numpy as np
from collections import namedtuple
from scipy.signal import savgol_filter

import hypers as hp

CRR = namedtuple('CRR', ['data', 'spectra', 'replace_method'])


def spectral_derivative_removal(X: 'hp.hparray', replace: str = 'nearest', window_length: int = 3) -> CRR:
    data = X.collapse()
    data_mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
    data_std = np.std(data, axis=1).reshape(data.shape[0], 1)

    # standardise prior to CR detection
    data -= data_mean
    data /= data_std

    # CR detection
    data = savgol_filter(data, window_length=window_length, polyorder=2, deriv=2, axis=1)

