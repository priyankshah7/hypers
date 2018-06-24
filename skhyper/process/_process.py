"""
Stores data in a custom class and generates attributes for other modules
"""

import warnings
import numpy as np
from scipy.signal import savgol_filter as _savgol

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

    normalize : bool
        Normalizes each spectra by subtracing the mean spectrum of the hyperspectral dataset.

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
    >>> X.ndim
    4
    >>>
    >>> X.n_features
    1024
    >>>
    >>> X.n_samples
    100000

    """
    def __init__(self, X, scale=True, normalize=False):
        self.data = X
        self._scale = scale
        self._normalize = normalize

        # Data properties
        self.shape = None
        self.ndim = None
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
        if self._normalize: self._data_normalization()
        self._data_mean()
        self._data_access()

    def _data_checks(self):
        if type(self.data) != np.ndarray:
            raise TypeError('Data must be a numpy array')

        self.shape = self.data.shape
        self.ndim = len(self.shape)

        if self.ndim != 3 and self.ndim != 4:
            raise TypeError('Data must be 3- or 4- dimensional.')

        if self.ndim == 3:
            self.n_samples = self.shape[0] * self.shape[1]
            self.n_features = self.shape[2]

            if not self.n_samples > self.n_features:
                # raise TypeError('The number of samples must be greater than the number of features')
                warnings.warn('n_samples (number of pixels) should be greater than n_features (spectral points)')

        elif self.ndim == 4:
            self.n_samples = self.shape[0] * self.shape[1] * self.shape[2]
            self.n_features = self.shape[3]

            if not self.n_samples > self.n_features:
                raise TypeError('The number of samples must be greater than the number of features')

    def _data_flatten(self):
        if self.ndim == 3:
            return np.reshape(self.data, (self.shape[0] * self.shape[1], self.shape[2]))

        elif self.ndim == 4:
            return np.reshape(self.data, (self.shape[0] * self.shape[1] * self.shape[2], self.shape[3]))

    def _data_mean(self):
        if self.ndim == 3:
            self.mean_image = np.squeeze(np.mean(self.data, 2))
            self.mean_spectrum = np.squeeze(np.mean(np.mean(self.data, 1), 0))

        elif self.ndim == 4:
            self.mean_image = np.squeeze(np.mean(self.data, 3))
            self.mean_spectrum = np.squeeze(np.mean(np.mean(np.mean(self.data, 2), 1), 0))

    def _data_scale(self):
        """ Scale the hyperspectral data

        Scales the hyperspectral data to between 0 and 1 for all positive data or
        -1 and 1 for positive and negative data.
        """
        self.data = self.data / np.max(np.abs(self.data))

    def _data_normalization(self):
        """Normalize the hyperspectral data

        Normalizes the hyperspectral data by subtracting the mean spectrum of the
        data from each pixel.
        """
        if self.ndim == 3:
            mean_spectrum = np.squeeze(np.mean(np.mean(self.data, 1), 0))

            for xpix in range(self.shape[0]):
                for ypix in range(self.shape[1]):
                    self.data[xpix, ypix, :] -= mean_spectrum

        elif self.ndim == 4:
            mean_spectrum = np.squeeze(np.mean(np.mean(np.mean(self.data, 2), 1), 0))

            for xpix in range(self.shape[0]):
                for ypix in range(self.shape[1]):
                    for zpix in range(self.shape[2]):
                        self.data[xpix, ypix, zpix, :] -= mean_spectrum

    def _data_access(self):
        self.image = _AccessImage(self.data, self.shape, self.ndim)
        self.spectrum = _AccessSpectrum(self.data, self.shape, self.ndim)

    def view(self):
        """ Hyperspectral viewer

        Opens a hyperspectral viewer with the hyperspectral array loaded (pyqt GUI)
        """
        hsiPlot(self)

    def smoothen(self, window_length=5, polyorder=3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0):
        """ Savitzky-Golay filter

        Applies a Savitzky-Golay filter to all the spectral components in the hyperspectral array.
        The data is modified directly in the object, i.e. a new object is not created
        (this is to limit memory usage).

        Parameters
        ----------
        window_length : int, optional (default: 5)
            The length of the filter window (i.e. the number of coefficients). `window_length`
            must be a positive odd integer.

        polyorder : int, optional (default: 3)
            The order of the polynomial used to fit the samples. `polyorder` must be less than
            `window_length`.

        deriv : int, optional (default: 0)
            The order of the derivative to compute. This must be a nonnegative intefer. The
            default it 0, which means to filter the data without differentiating.

        delta : float, optional (default: 1.0)
            The spacing of the samples to which the data will be applied. This is only used
            if `deriv`>0.

        axis : int, optional (default: -1)
            The axis of the array along which the filter is to be applied

        mode : str, optional (default: 'interp')
            Must be ‘mirror’, ‘constant’, ‘nearest’, ‘wrap’ or ‘interp’. This determines
            the type of extension to use for the padded signal to which the filter is
            applied. When mode is ‘constant’, the padding value is given by `cval`.
            See the Notes for more details on ‘mirror’, ‘constant’, ‘wrap’, and
            ‘nearest’. When the ‘interp’ mode is selected (the default), no
            extension is used. Instead, a degree `polyorder` polynomial is fit to the
            last `window_length` values of the edges, and this polynomial is used to
            evaluate the last `window_length` // 2 output values.

        cval : scalar, optional (default: 0.0)
            Value to fill past the edges of the input if `mode` is 'constant'.

        """
        if self.ndim == 3:
            for xpix in range(self.shape[0]):
                for ypix in range(self.shape[1]):
                    self.data[xpix, ypix, :] = _savgol(self.data[xpix, ypix, :], window_length=window_length,
                                                       polyorder=polyorder, deriv=deriv, delta=delta,
                                                       axis=axis, mode=mode, cval=cval)

        elif self.ndim == 4:
            for xpix in range(self.shape[0]):
                for ypix in range(self.shape[1]):
                    for zpix in range(self.shape[2]):
                        self.data[xpix, ypix, zpix, :] = _savgol(self.data[xpix, ypix, :], window_length=window_length,
                                                                 polyorder=polyorder, deriv=deriv, delta=delta,
                                                                 axis=axis, mode=mode, cval=cval)

        self.update()

    def flatten(self):
        """Flatten the hyperspectral data

        Flattens the hyperspectral data from 3d/4d to 2d by unravelling the pixel order.

        Returns
        -------
        X_flattened : array, shape (x*y*(z), n_features)
            A flattened version of the hyperspectral array

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
