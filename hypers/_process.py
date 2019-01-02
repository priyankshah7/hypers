"""
Stores data in a custom class and generates attributes for other modules
"""
import numpy as np
from typing import Tuple, Union, List
from sklearn.preprocessing import scale, robust_scale, normalize

from hypers._preprocessing import _data_preprocessing
from hypers._learning import _data_cluster, _vca, _ucls, _data_decomposition, _data_scree, _data_mixture
from hypers._tools import _data_smoothen, _data_mean, _data_checks, _data_access
from hypers._tools import PreprocessType, ClusterType, DecomposeType, MixtureType
from hypers._view import hsiPlot


class Dataset:
    """ Dataset data structure

    Attributes
    ----------
    data: np.ndarray
        The raw hyperspectral data

    shape: tuple
        Shape of the hyperspectral data

    ndim: int
        Number of dimensions of the data (3 or 4)

    n_features: int
        Number of spectral bands

    n_samples: int
        Total number of pixels

    mean_spectrum: np.ndarray
        An array of the mean spectrum of the data

    mean_image: np.ndarray
        An array of the mean image of the data

    image: np.ndarray
        An array of the image averaged over the specified spectral bands

    spectrum: np.ndarray
        An array of the spectrum averaged over the specified pixels

    smoothing: str
        Smoothing to use. Can be either:

        - savitzky_golay
        - gaussian

        Default is 'savitzky_golay'.

    """
    def __init__(self, data: np.ndarray) -> None:
        self.data = np.squeeze(data)

        # Data properties
        self.shape = None
        self.ndim = None
        self.n_features = None
        self.n_samples = None
        self.smoothing = 'savitzky_golay'
        self._scaled = False
        self._robust_scaled = False
        self._normalized = False

        # Hyperspectral image/spectrum
        self.image = None
        self.spectrum = None
        self.mean_image = None
        self.mean_spectrum = None

        # sklearn
        self.mdl_preprocess = None
        self.mdl_decompose = None
        self.mdl_cluster = None
        self.mdl_mixture = None

        self.update()

    def __getitem__(self, key: tuple) -> np.ndarray:
        return self.data[key]

    def __setitem__(self, key: tuple, value: Union[int, float, np.ndarray]) -> None:
        self.data[key] = value
        self.update()

    def __str__(self) -> str:
        str_sklearn = [0] * 4
        cls_sklearn = [self.mdl_preprocess, self.mdl_decompose, self.mdl_cluster, self.mdl_mixture]

        for sktype in range(4):
            if type(cls_sklearn[sktype]) is not None.__class__:
                str_sklearn[sktype] = cls_sklearn[sktype].__class__.__name__
            else:
                str_sklearn[sktype] = 'None'

        descr = ('---Dataset class---\n' +
                 'Type: ' + str(self.ndim) + '-dimensional hyperspectral data\n' +
                 'Image dimension: ' + str(self.shape[:-1]) + ' | ' +
                 'Spectral bands: ' + str(self.shape[-1]) + '\n' +
                 'Scaled: ' + str(self._scaled) + ' | ' +
                 'Scaled (robust): ' + str(self._robust_scaled) + '\n'+
                 'Normalized: ' + str(self._normalized) + '\n' +
                 'Preprocessing (sklearn): ' + str_sklearn[0] + '\n' +
                 'Decomposition (sklearn): ' + str_sklearn[1] + '\n' +
                 'Cluster (sklearn): ' + str_sklearn[2] + '\n' +
                 'Gaussian mixture models (sklearn): ' + str_sklearn[3])

        return descr

    def __truediv__(self, var: Union[int, float, np.ndarray]) -> 'Dataset':
        if type(var) in (int, float):
            for _val in np.ndenumerate(self.data):
                self.data[_val[0]] /= var

        elif type(var) == np.ndarray and var.ndim == 1 and var.shape[0] == self.shape[-1]:
            for _val in np.ndindex(self.shape[:-1]):
                self.data[_val] /= var

        elif type(var) == np.ndarray and var.ndim == self.ndim-1 and var.shape == self.shape[:-1]:
            for _band in range(self.shape[-1]):
                self.data[..., _band] /= var

        else:
            raise TypeError('Can only divide by an integer, float or spectral array of shape (' + str(self.shape[-1])
                            + ') or ' + str(self.shape[:-1]))

        self.update()
        return self

    def __mul__(self, var: Union[int, float, np.ndarray]) -> 'Dataset':
        if type(var) in (int, float):
            for _val in np.ndenumerate(self.data):
                self.data[_val[0]] *= var

        elif type(var) == np.ndarray and var.ndim == 1 and var.shape[0] == self.shape[-1]:
            for _val in np.ndindex(self.shape[:-1]):
                self.data[_val] *= var

        elif type(var) == np.ndarray and var.ndim == self.ndim-1 and var.shape == self.shape[:-1]:
            for _band in range(self.shape[-1]):
                self.data[..., _band] *= var

        else:
            raise TypeError('Can only multiply by an integer, float or spectral array of shape (' + str(self.shape[-1])
                            + ') or ' + str(self.shape[:-1]))

        self.update()
        return self

    def __add__(self, var: Union[int, float, np.ndarray]) -> 'Dataset':
        if type(var) in (int, float):
            for _val in np.ndenumerate(self.data):
                self.data[_val[0]] += var

        elif type(var) == np.ndarray and var.ndim == 1 and var.shape[0] == self.shape[-1]:
            for _val in np.ndindex(self.shape[:-1]):
                self.data[_val] += var

        elif type(var) == np.ndarray and var.ndim == self.ndim-1 and var.shape == self.shape[:-1]:
            for _band in range(self.shape[-1]):
                self.data[..., _band] += var

        else:
            raise TypeError('Can only add by an integer, float or spectral array of shape (' + str(self.shape[-1])
                            + ') or ' + str(self.shape[:-1]))

        self.update()
        return self

    def __sub__(self, var: Union[int, float, np.ndarray]) -> 'Dataset':
        if type(var) in (int, float):
            for _val in np.ndenumerate(self.data):
                self.data[_val[0]] -= var

        elif type(var) == np.ndarray and var.ndim == 1 and var.shape[0] == self.shape[-1]:
            for _val in np.ndindex(self.shape[:-1]):
                self.data[_val] -= var

        elif type(var) == np.ndarray and var.ndim == self.ndim-1 and var.shape == self.shape[:-1]:
            for _band in range(self.shape[-1]):
                self.data[..., _band] -= var

        else:
            raise TypeError('Can only subtract by an integer, float or spectral array of shape (' + str(self.shape[-1])
                            + ') or ' + str(self.shape[:-1]))

        self.update()
        return self

    def update(self) -> None:
        """Update the stored data and class properties"""
        _data_checks(self)
        _data_mean(self)
        _data_access(self)

    def view(self) -> None:
        """Open the hyperspectral viewer GUI"""
        hsiPlot(self)

    def smoothen(self, **kwargs) -> None:
        """Smoothen the hyperspectral data

        Parameters
        ----------
        **kwargs
            Keyword arguments for either of the following (depending on what the smoothing attribute
            has been set to):

            - ``scipy.signal.savgol_filter``
            - ``scipy.ndimage.filters.gaussian_filter``
        """
        _data_smoothen(self, **kwargs)
        self.update()

    def scale(self, with_mean: bool = True,
              with_std: bool = True) -> None:
        """Standardize the hyperspectral data

        Parameters
        ----------
        with_mean: bool
            If True, center the data before scaling

        with_std: bool
            If True, scale the data to unit variance
        """
        self.data = scale(self.flatten(), axis=0, with_mean=with_mean, with_std=with_std).reshape(self.shape)
        self._scaled = True
        self.update()

    def robust_scale(self, with_centering: bool = True,
                     with_scaling: bool = True,
                     quantile_range: tuple = (25.0, 75.0)) -> None:
        """Standardize the hyperspectral data (robust from outliers)

        Parameters
        ----------
        with_centering: bool
            If True, center the data before scaling

        with_scaling: bool
            If True, scale the data to unit variance

        quantile_range: tuple
            Quantile range used.
        """
        self.data = robust_scale(self.flatten(), with_centering=with_centering, with_scaling=with_scaling,
                                 quantile_range=quantile_range).reshape(self.shape)
        self._robust_scaled = True
        self.update()

    def normalize(self, norm: str = 'l2') -> None:
        """Normalize the hyperspectral data

        Parameters
        ----------
        norm: str ('l1', 'l2', 'max')
            The norm to use to normalize each spectrum.
        """
        self.data = normalize(self.flatten(), axis=0, norm=norm).reshape(self.shape)
        self._normalized = True
        self.update()

    def flatten(self) -> np.ndarray:
        """ Returns flattened array

        Returns
        -------
        x_flat: np.ndarray
            Flattened 2d array of the stored hyperspectral data
        """
        return np.reshape(self.data, (np.prod(self.shape[:-1]), self.shape[-1]))

    def scree(self, plot: bool = False,
              return_arrs: bool = True) -> np.ndarray:
        """ Returns PCA scree

        Parameters
        ----------
        plot: bool
            If True, will plot the array

        return_arrs: bool
            If True, will return the array

        Returns
        -------
        scree: np.ndarray
            An array of the PCA scree
        """
        return _data_scree(self, plot=plot, return_arrs=return_arrs)

    def preprocess(self, mdl: PreprocessType) -> None:
        """ Preprocess data

        Parameters
        ----------
        mdl: PreprocessType
            Accepts a ``scikit-learn`` preprocessing class
        """
        _data_preprocessing(self, mdl)

    def decompose(self, mdl: DecomposeType,
                  plot: bool = False,
                  return_arrs: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose data

        Parameters
        ----------
        mdl: DecomposeType
            Accepts a ``scikit-learn`` decomposition class

        plot: bool
            If True, will return a plot of the images/spectra of the components

        return_arrs: bool
            If True, will return the arrays of the images/spectra

        Returns
        -------
        ims: np.ndarray
            An array of images (size: x, y, (z), n_components)

        specs: np.ndarray
            An array of spectra (size: spectra, n_components)
        """

        return _data_decomposition(self, mdl, return_arrs=return_arrs, plot=plot)

    def cluster(self, mdl: ClusterType,
                decomposed: bool = False,
                pca_comps: int = 4,
                plot: bool = False,
                return_arrs: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster data

        Parameters
        ----------
        mdl: ClusterType
            Accepts a ``scikit-learn`` cluster class

        decomposed: bool
            Whether to perform PCA on the data prior to clustering

        pca_comps: int
            If decomposed=True, this specifies the number of principal components to use

        plot: bool
            If True, will return a plot of the labels/spectra of the clusters

        return_arrs: bool
            If True, will return the arrays of the labels/spectra of the clusters

        Returns
        -------
        ims: np.ndarray
            An array of the labels (size: x, y, (z))

        specs: np.ndarray
            An array of spectra (size: spectra, n_clusters)
        """

        return _data_cluster(self, mdl, decomposed=decomposed, pca_comps=pca_comps, plot=plot, return_arrs=return_arrs)

    def mixture(self, mdl: MixtureType,
                plot: bool = False,
                return_arrs: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Gaussian mixture models

        Parameters
        ----------
        mdl: MixtureType
            Accepts a ``scikit-learn`` mixture class

        plot: bool
            If True, will return a plot of the labels/spectra of the components

        return_arrs: bool
            If True, will return the arrays of the labels/spectra of the components

        Returns
        -------
        labels: np.ndarray
            An array of the labels

        spectra: np.ndarray
            An array of the spectra of the components
        """

        return _data_mixture(self, mdl, plot=plot, return_arrs=return_arrs)

    def vca(self, n_components: int = 4,
            plot: bool = False,
            return_arrs: bool = True) -> Tuple[np.ndarray, List[int]]:
        """Vertex component analysis

        Parameters
        ----------
        n_components: int
            Number of pure components to find

        plot: bool
            If True, will return a plot of the pure spectra

        return_arrs: bool
            If True, will return an array of the pure spectra and a list of tuples of the coordinates of the
            pure pixels

        Returns
        -------
        spectra: np.ndarray
            An array of the spectra of the pure pixels

        coords: np.ndarray
            A list of tuples of the coordinates of the pure pixels
        """

        Ae, indices = _vca(self, n_components=n_components, plot=plot, return_arrs=return_arrs)
        return Ae, indices

    def abundance(self, spectra: np.ndarray,
                  plot: bool = False,
                  return_arrs: bool = True) -> np.ndarray:
        """Abundance map with least-squares fitting

        Parameters
        ----------
        spectra: np.ndarray
            Spectra to perform fitting with

        plot: bool
            If True, will return an image of the abundance map

        return_arrs: bool
            If True, will return an array of the abundance map

        Returns
        -------
        im: np.ndarray
            An array of the abundance map
        """

        return _ucls(self, spectra=spectra, plot=plot, return_arrs=return_arrs)
