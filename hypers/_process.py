"""
Stores data in a custom class and generates attributes for other modules
"""
import numpy as np
from typing import Tuple, Union
from sklearn.preprocessing import (
    MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler,
    StandardScaler, Normalizer
)
from sklearn.decomposition import (
    PCA, FastICA, IncrementalPCA, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
    FactorAnalysis, NMF, LatentDirichletAllocation
)
from sklearn.cluster import (
    KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN
)
from hypers._preprocessing import _data_preprocessing, _data_scale
from hypers._tools._smoothen import _data_smoothen
from hypers._learning._cluster import _data_cluster
from hypers._learning._decomposition import _data_decomposition, _data_scree
from hypers._tools._plotting import _data_plotting
from hypers._tools._types import PreprocessType, ClusterType, DecomposeType
from hypers._tools._update import (
    _data_access, _data_checks, _data_mean
)
from hypers._view import hsiPlot


class Dataset:
    """Dataset structure to store the hyperspectral array.

    Parameters
    ----------
    X : array, dimensions (3 or 4)
        The hyperspectral data. It should be 3- or 4-dimensional in the form:
            X_3d = [x, y, spectrum] or
            X_4d = [x, y, z, spectrum]

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

    image : array, shape(x, y, (z))
        Returns the image averaged over the selected spectral range

    spectrum : array, shape(n_features)
        Returns the spectrum averaged over the selected pixels

    mean_image : array, shape(x, y, (z))
        Returns the image averaged over the entire spectral range

    mean_spectrum : array, shape(n_features)
        Returns the spectrum averaged over all the pixels

    """
    def __init__(self, data: np.ndarray) -> None:
        self.data = data

        # Data properties
        self.shape = None
        self.ndim = None
        self.n_features = None
        self.n_samples = None
        self.smoothing = 'savitzky_golay'

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

    def __getitem__(self, key) -> np.ndarray:
        return self.data[key]

    def __setitem__(self, key, value) -> None:
        self.data[key] = value
        self.update()

    def __truediv__(self, var: Union[int, float, np.ndarray]) -> 'Dataset':
        if type(var) in (int, float):
            for _val in np.ndenumerate(self.data):
                self.data[_val[0]] /= var

        elif type(var) == np.ndarray and var.ndim == 1 and var.shape[0] == self.shape[-1]:
            for _val in np.ndindex(self.shape[:-1]):
                self.data[_val] /= var

        else:
            raise TypeError('Can only divide by an integer, float or spectral array')

        self.update()
        return self

    def __mul__(self, var: Union[int, float, np.ndarray]) -> 'Dataset':
        if type(var) in (int, float):
            for _val in np.ndenumerate(self.data):
                self.data[_val[0]] *= var

        elif type(var) == np.ndarray and var.ndim == 1 and var.shape[0] == self.shape[-1]:
            for _val in np.ndindex(self.shape[:-1]):
                self.data[_val] *= var

        else:
            raise TypeError('Can only multiply by an integer, float or spectral array')

        self.update()
        return self

    def __add__(self, var: Union[int, float, np.ndarray]) -> 'Dataset':
        if type(var) in (int, float):
            for _val in np.ndenumerate(self.data):
                self.data[_val[0]] += var

        elif type(var) == np.ndarray and var.ndim == 1 and var.shape[0] == self.shape[-1]:
            for _val in np.ndindex(self.shape[:-1]):
                self.data[_val] += var

        else:
            raise TypeError('Can only add with an integer, float or spectral array')

        self.update()
        return self

    def __sub__(self, var: Union[int, float, np.ndarray]) -> 'Dataset':
        if type(var) in (int, float):
            for _val in np.ndenumerate(self.data):
                self.data[_val[0]] -= var

        elif type(var) == np.ndarray and var.ndim == 1 and var.shape[0] == self.shape[-1]:
            for _val in np.ndindex(self.shape[:-1]):
                self.data[_val] -= var

        else:
            raise TypeError('Can only subtract by an integer, float or spectral array')

        self.update()
        return self

    def update(self) -> None:
        _data_checks(self)
        _data_mean(self)
        _data_access(self)

    def view(self) -> None:
        hsiPlot(self)

    def smoothen(self, **kwargs) -> None:
        _data_smoothen(self, **kwargs)
        self.update()

    def flatten(self) -> np.ndarray:
        return np.reshape(self.data, (np.prod(self.shape[:-1]), self.shape[-1]))

    def scree(self) -> np.ndarray:
        return _data_scree(self)

    def preprocess(self, mdl: PreprocessType,
                   scale_features: bool = True) -> None:

        if scale_features:
            _data_scale(self)
        _data_preprocessing(self, mdl)

    def decompose(self, mdl: DecomposeType,
                  return_arrs: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        return _data_decomposition(self, mdl, return_arrs)

    def cluster(self, mdl: ClusterType,
                decomposed: bool = False,
                pca_comps: int = 4,
                return_arrs: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        return _data_cluster(self, mdl, decomposed, pca_comps, return_arrs)

    def plot(self, kind: str = 'both',
             target: str = 'data',
             figsize: str = None):

        return _data_plotting(self, kind=kind, target=target, figsize=figsize)
