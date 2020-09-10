"""
Extends functionality of np.ndarray for hyperspectral data
"""
import numpy as np
from typing import Union
from pathlib import Path
from scipy.signal import savgol_filter

from hypers.exceptions import DimensionError
from hypers.core.abundance import ucls, nnls
from hypers.plotting.view import hsiPlot


class hparray(np.ndarray):
    """
    Extend functionality of a numpy array for hyperspectral data

    The usual `numpy.ndarray` attributes and methods are available
    as well as some additional ones that extend functionality.

    Parameters
    ----------
    input_array: Union[list, np.ndarray]
        The array to convert. This should either be a 2d/3d/4d
        numpy array (type `np.ndarray`) or list.

    Attributes
    ----------
    mean_image: np.ndarray
        Provides the mean image by averaging across the spectral
        dimension. e.g. if the shape of the original array is
        (100, 100, 512), then the image dimension shape is (100, 100)
        and the spectral dimension shape is (512,). So the mean
        image will be an array of shape (100, 100).

    mean_spectrum: np.ndarray
        Provides the mean spectrum by averaging across the image
        dimensions. e.g. if the shape of the original array is
        (100, 100, 512), then the image dimension shape is (100, 100)
        and the spectral dimension shape is (512,). So the mean
        spectrum will be an array of shape (512,).
    """
    def __new__(cls, input_array: Union[np.ndarray, 'hparray']):
        obj = np.asarray(input_array)#.view(cls)
        if obj.ndim > 1:
            return obj.view(cls)
        else:
            return obj
        # return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        else:
            if self.ndim > 1:
                self._data_access()
            # else:
            #     self._dimension_error()

    def __array_wrap__(self, output_array, context=None):
        return np.ndarray.__array_wrap__(self, output_array)

    def _data_access(self):
        self.image = _AccessImage(self)
        self.spectrum = _AccessSpectrum(self)

    @staticmethod
    def _dimension_error(error: str = None):
        if error is None:
            error = 'Number of dimensions must be greater than 2'
        raise DimensionError(error)

    def abundance(self, x_fit: np.ndarray, method: str = 'nnls') -> np.ndarray:
        """
        Abundance mapping using a least-squares fitting routine.

        Parameters
        ----------
        x_fit: np.ndarray
            Array of spectra to fit to the hyperspectral data. Must be of
            size (nspectral, n_fit) where n_fit is the number of spectra
            provided to fit.
        method: str
            Type of least-squares fitting to use. Can be either 'ucls'
            (unconstrained least-squares) or 'nnls' (non-negative
            least squares).

        Returns
        -------
        np.ndarray
            Array of images with an image per spectrum to fit. Array has
            size of (nspatial, n_fit).
        """
        if x_fit.ndim == 1:
            x_fit = x_fit.reshape(x_fit.shape[0], 1)

        assert x_fit.shape[0] == self.nspectral

        if method == 'nnls':
            return nnls(self, x_fit)
        elif method == 'ucls':
            return ucls(self, x_fit)
        else:
            raise ValueError('method argument for the abundance method must '
                             'be either "ucls" or "nnls"')

    def collapse(self) -> np.ndarray:
        """
        Collapse the array into a 2d array

        Collapses the array into a 2d array, where the first dimension
        is the collapsed image dimensions and the second dimension is
        the spectral dimension.

        Returns
        -------
        np.ndarray
            The collapsed 2d numpy array.

        Examples
        --------
        >>> import numpy as np
        >>> import hypers as hp
        >>> data = np.random.rand(40, 30, 1000)
        >>> x = hp.hparray(data)
        >>> collapsed = x.collapse()
        >>> collapsed.shape
        (1200, 1000)
        """
        return np.asarray(np.reshape(self, (np.prod(self.shape[:-1]), self.shape[-1])))

    def smoothen(self, method: str = 'savgol', **kwargs) -> 'hparray':
        """
        Returns smoothened hp.hparray

        Parameters
        ----------
        method: str
            Method to use to smooth the array. Default is 'savgol'.
            + 'savgol': Savitzky-Golay filter.

        **kwargs
            Keyword arguments for the relevant method used.
            + method='savgol'
                kwargs for the `scipy.signal.savgol_filter` implementation

        Returns
        -------
        hp.hparray
            The smoothened array with the same dimensions as the original
            array.
        """
        smooth_array = self.copy()
        for index in np.ndindex(self.shape[:-1]):
            if method == 'savgol':
                smooth_array[index] = savgol_filter(smooth_array[index], **kwargs)
            else:
                raise ValueError
        return smooth_array

    def plot(self) -> None:
        """
        Interactive plotting to interact with hyperspectral data

        Note that at the moment only the 'pyqt' backend has been implemented. This means that
        PyQt is required to be installed and when this method is called, a separate window generated
        by PyQt will pop up. It is still possible to use this in a Jupyter environment, however the
        cell that calls this method will remain frozen until the window is closed.
        """
        hsiPlot(self)

    def save(self, filename: Union[str, Path]) -> None:
        """
        Save hyperspectral data to file using numpy's save function.

        Parameters
        ----------
        filename: str or pathlib.Path
            Filename with extension of desired file format. If no extension is given then it will
            automatically be saved to the npy file format.
        """
        np.save(filename, self)

    @property
    def mean_image(self) -> np.ndarray:
        """
        Returns the mean image of the hyperspectral array

        Returns
        -------
        np.ndarray:
            Mean image of the hyperspectral array
        """
        return np.array(np.squeeze(np.mean(self, self.ndim - 1)))

    @property
    def mean_spectrum(self) -> np.ndarray:
        """
        Returns the mean spectrum of the hyperspectral array

        Returns
        -------
        np.ndarray:
            Mean spectrum of the hyperspectral array
        """
        return np.array(np.squeeze(np.mean(self, tuple(range(self.ndim - 1)))))

    @property
    def nsamples(self) -> int:
        """
        Returns the number of samples (total number of spatial pixels) in the datasets

        Returns
        -------
        int:
            Total number of samples
        """
        return int(np.prod(self.shape[:-1]))

    @property
    def nspatial(self) -> tuple:
        """
        Returns the shape of the spatial dimensions

        Returns
        -------
        tuple:
            Tuple of the shape of the spatial dimensions
        """
        return self.shape[:-1]

    @property
    def nspectral(self) -> int:
        """
        Returns the number of spectral bands in the datasets

        Returns
        -------
        int:
            Size of the spectral dimension
        """
        return self.shape[-1]


class _AccessImage:
    def __init__(self, X):
        self.X = X

    def __getitem__(self, item):
        if isinstance(item, tuple):
            raise IndexError('Can only pass in 1 index (same as number of spectral dimension)')

        if isinstance(item, int):
            return np.array(np.squeeze(self.X.data[..., item]))
        elif isinstance(item, slice):
            return np.array(np.squeeze(np.mean(self.X.data[..., item], self.X.ndim - 1)))


class _AccessSpectrum:
    def __init__(self, X):
        self.X = X

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            item = (item,)
        elif not len(item) == self.X.ndim - 1:
            raise IndexError('Must pass in the same number of indicies as number of image dimensions')

        if isinstance(item, tuple) and all(isinstance(nitem, int) for nitem in item):
            return np.array(np.squeeze(self.X.data[item]))
        elif isinstance(item, tuple) and all(isinstance(nitem, slice) for nitem in item):
            return np.array(np.squeeze(np.mean(self.X.data[item], tuple(range(self.X.ndim - 1)))))
