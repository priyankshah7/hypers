import warnings
import numpy as np


def _data_checks(X):
    if type(X.data) != np.ndarray:
        raise TypeError('Data must be a numpy array')

    X.shape = X.data.shape
    X.ndim = len(X.shape)

    if X.ndim != 3 and X.ndim != 4:
        raise TypeError('Data must be 3- or 4- dimensional.')

    if X.ndim == 3:
        X.n_samples = X.shape[0] * X.shape[1]
        X.n_features = X.shape[2]

        if not X.n_samples > X.n_features:
            warnings.warn('n_samples (number of pixels) should be greater than n_features (spectral points)')

    elif X.ndim == 4:
        X.n_samples = X.shape[0] * X.shape[1] * X.shape[2]
        X.n_features = X.shape[3]

        if not X.n_samples > X.n_features:
            warnings.warn('n_samples (number of pixels) should be greater than n_features (spectral points)')


def _data_mean(X):
    if X.ndim == 3:
        X.mean_image = np.squeeze(np.mean(X.data, 2))
        X.mean_spectrum = np.squeeze(np.mean(np.mean(X.data, 1), 0))

    elif X.ndim == 4:
        X.mean_image = np.squeeze(np.mean(X.data, 3))
        X.mean_spectrum = np.squeeze(np.mean(np.mean(np.mean(X.data, 2), 1), 0))


def _data_access(X):
    X.image = _AccessImage(X.data, X.shape, X.ndim)
    X.spectrum = _AccessSpectrum(X.data, X.shape, X.ndim)


class _AccessImage:
    def __init__(self, X, shape, ndim):
        self.data = X
        self.shape = shape
        self.ndim = ndim

    def __getitem__(self, item):
        return np.squeeze(np.mean(self.data[item], self.ndim - 1))


class _AccessSpectrum:
    def __init__(self, X, shape, ndim):
        self.data = X
        self.shape = shape
        self.ndim = ndim

    def __getitem__(self, item):
        if self.ndim == 3:
            return np.squeeze(np.mean(np.mean(self.data[item], 1), 0))

        elif self.ndim == 4:
            return np.squeeze(np.mean(np.mean(np.mean(self.data[item], 2), 1), 0))
