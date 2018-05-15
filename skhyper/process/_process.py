"""
Stores data in a custom class and generates attributes for other modules
"""

import warnings
import numpy as np

from skhyper.view import hsiPlot
from sklearn.decomposition import PCA


class Process:
    """
    Process object to store the hyperspectral array.

    Parameters
    ----------
    X : array, dimensions (3 or 4)
        The hyperspectral data. It should be 3- or 4-dimensional in the form:
            X_3d = [x, y, spectrum] or
            X_4d = [x, y, z, spectrum]

    scale : bool
        Scales the spectra to either between {0, 1} or {-1, 1} depending on presence of negative values.

    Attributes
    ----------
    shape : array
        Returns the shape of the hyperspectral array

    n_dimension : int
        Returns the number of dimensions of the hyperspectral array (3 or 4)

    n_features : int
        Returns the number of spectral points (features)

    n_samples : int
        Returns the total number of pixels in the hyperspectral array

    flat : array, dimension (2)
        Returns a flattened 2-d version of the hyperspectral array

    image : array, shape(x, y, (z))
        Returns the image averaged over the selected spectral range

    spectrum : array, shape(n_features)
        Returns the spectrum averaged over the selected pixels

    mean_image : array, shape(x, y, (z))
        Returns the image averaged over the entire spectral range

    mean_spectrum : array, shape(n_features)
        Returns the spectrum averaged over all the pixels


    Examples
    --------
    >>> import numpy as np
    >>> from skhyper.process import Process
    >>>
    >>> test_data = np.random.rand(100, 100, 10, 1024)
    >>> X = Process(test_data, scale=True)
    >>>
    >>> X.n_dimension
    4
    >>>
    >>> X.n_features
    1024
    >>>
    >>> X.n_samples
    100000

    """
    def __init__(self, X, scale=True):
        self.data = X
        self._scale = scale

        # Data properties
        self.shape = None
        self.n_dimension = None
        self.n_features = None
        self.n_samples = None

        # Hyperspectral image/spectrum
        self.image = None
        self.spectrum = None
        self.mean_image = None
        self.mean_spectrum = None

        self.update()

    def __getitem__(self, item):
        return self.data[item]

    def update(self):
        """ Update properties of the hyperspectral array

        This should be called whenever `X.data` is directly modified to update the attributes
        of the `X` object.

        """
        # Perform data operations
        self._data_checks()
        if self._scale: self._data_scale()
        self._data_mean()
        self._data_access()

    def _data_checks(self):
        if type(self.data) != np.ndarray:
            raise TypeError('Data must be a numpy array')

        self.shape = self.data.shape
        self.n_dimension = len(self.shape)

        if self.n_dimension != 3 and self.n_dimension != 4:
            raise TypeError('Data must be 3- or 4- dimensional.')

        if self.n_dimension == 3:
            self.n_samples = self.shape[0] * self.shape[1]
            self.n_features = self.shape[2]

            if not self.n_samples > self.n_features:
                # raise TypeError('The number of samples must be greater than the number of features')
                warnings.warn('n_samples (number of pixels) should be greater than n_features (spectral points)')

        elif self.n_dimension == 4:
            self.n_samples = self.shape[0] * self.shape[1] * self.shape[2]
            self.n_features = self.shape[3]

            if not self.n_samples > self.n_features:
                raise TypeError('The number of samples must be greater than the number of features')

    def _data_flatten(self):
        if self.n_dimension == 3:
            return np.reshape(self.data, (self.shape[0] * self.shape[1], self.shape[2]))

        elif self.n_dimension == 4:
            return np.reshape(self.data, (self.shape[0] * self.shape[1] * self.shape[2], self.shape[3]))

    def _data_mean(self):
        if self.n_dimension == 3:
            self.mean_image = np.squeeze(np.mean(self.data, 2))
            self.mean_spectrum = np.squeeze(np.mean(np.mean(self.data, 1), 0))

        elif self.n_dimension == 4:
            self.mean_image = np.squeeze(np.mean(self.data, 3))
            self.mean_spectrum = np.squeeze(np.mean(np.mean(np.mean(self.data, 2), 1), 0))

    def _data_scale(self):
        self.data = self.data / np.abs(np.max(self.data))

    def _data_access(self):
        self.image = _AccessImage(self.data, self.shape, self.n_dimension)
        self.spectrum = _AccessSpectrum(self.data, self.shape, self.n_dimension)

    def view(self):
        """ Hyperspectral viewer

        Opens a hyperspectral viewer with the hyperspectral array loaded (pyqt GUI)
        """
        hsiPlot(self)

    def flatten(self):
        """Flatten the hyperspectral data

        Flattens the hyperspectral data from 3d/4d to 2d by unravelling the pixel order.
        """
        return self._data_flatten()

    def scree(self):
        """Returns the array for the scree plot

        Returns the scree plot from `PCA` as an array.

        Returns
        -------
        scree : array, shape (n_features,)

        """
        mdl = PCA()
        mdl.fit_transform(self.flatten())
        scree = mdl.explained_variance_ratio_

        return scree


class _AccessImage:
    def __init__(self, X, shape, n_dimension):
        self.data = X
        self.shape = shape
        self.n_dimension = n_dimension

    def __getitem__(self, item):
        if self.n_dimension == 3:
            return np.squeeze(np.mean(self.data[item], 2))

        elif self.n_dimension == 4:
            return np.squeeze(np.mean(self.data[item], 3))


class _AccessSpectrum:
    def __init__(self, X, shape, n_dimension):
        self.data = X
        self.shape = shape
        self.n_dimension = n_dimension

    def __getitem__(self, item):
        if self.n_dimension == 3:
            return np.squeeze(np.mean(np.mean(self.data[item], 1), 0))

        elif self.n_dimension == 4:
            return np.squeeze(np.mean(np.mean(np.mean(self.data[item], 2), 1), 0))
