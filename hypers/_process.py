"""
Stores data in a custom class and generates attributes for other modules
"""
import warnings
import numpy as np

from hypers._preprocessing import _data_preprocessing
from hypers._tools._smoothen import _data_smoothen
from hypers._learning._cluster import _data_cluster
from hypers._learning._decomposition import _data_decomposition, _data_scree
from hypers._view import hsiPlot


class Dataset:
    """
    Process object to store the hyperspectral array.

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


    Examples
    --------
    >>> import numpy as np
    >>> import hypers as hp
    >>>
    >>> test_data = np.random.rand(100, 100, 10, 1024)
    >>> X = hp.Dataset(test_data, scale=True)
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
    def __init__(self, X):
        # TODO Store in numpy memmap
        self.data = X

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

    def _data_access(self):
        self.image = _AccessImage(self.data, self.shape, self.ndim)
        self.spectrum = _AccessSpectrum(self.data, self.shape, self.ndim)

    def view(self):
        """ Hyperspectral viewer

        Opens a hyperspectral viewer with the hyperspectral array loaded (pyqt GUI)
        """
        hsiPlot(self)

    def smoothen(self, **kwargs):
        """ Data smoothening
        
        Smoothens all spectra in the dataset using one of the following:

        - Savitzky-Golay filter (scipy.signal.savgol_filter)
        - Gaussian filter (scipy.ndimage.filters.gaussian_filter)

        Savitzky-Golay is chosen by default. To choose the Gaussian filter, 
        set `smoothing='gaussian_filter'`. e.g.

        >>> import numpy as np
        >>> import hypers as hp
        >>> data = np.random.rand(50, 50, 100)
        >>> X = hp.Dataset(data)
        >>> X.smoothing
        'savitzky_golay'
        >>> X.smoothing = 'gaussian_filter'
        >>> X.smoothen()

        Parameters
        ----------
        **kwargs : Savitsky-Golay or Gaussian filter parameters

        """

        _data_smoothen(self, **kwargs)
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
        """ Scree plot
        
        Returns the scree plot by applying PCA. Useful to understand the contribution
        of each principal component to the total variance in the dataset.
        
        Returns
        -------
        scree : np.ndarray (n_features,)
        """

        return _data_scree(self)

    def preprocess(self, mdl):
        """ Preprocess stored dataset
        
        Preprocess the stored dataset using the following preprocessing classes from 
        scikit-learn.

        NOTE:
        This applies the preprocessing step to the features (i.e. spectra), not to the spatial 
        components.

        - MaxAbsScaler
        - MinMaxScaler
        - PowerTransformer
        - QuantileTransformer
        - RobustScaler
        - StandardScaler
        
        Parameters
        ----------
        mdl : object
            scikit-learn preprocessing class

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.preprocessing import StandardScaler
        >>> import hypers as hp
        >>> data = np.random.rand(50, 50, 100)
        >>> X = hp.Dataset(data)
        >>> X.preprocess(mdl=StandardScaler())
        """
        _data_preprocessing(self, mdl)

    def decompose(self, mdl):
        """ Dimensionality reduction
        
        Apply one of the following scikit-learn dimensionality reduction techniques to the
        stored dataset:

        - PCA
        - FastICA
        - IncrementalPCA
        - TruncatedSVD
        - DictionaryLearning
        - MiniBatchDictionaryLearning
        - FactorAnalysis
        - NMF
        - LatentDirichletAllocation
        
        Parameters
        ----------
        mdl : object
            scikit-learn decomposition class
        
        Returns
        -------
        ims : np.ndarray (n_samples, n_components)
            Images of the n_components number of principal components

        spcs : np.ndarray (n_features, n_components)
            Spectra of the n_components number of principal components

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.decomposition import PCA
        >>> import hypers as hp
        >>> data = np.random.rand(50, 50, 100)
        >>> X = hp.Dataset(data)
        >>> ims, spcs = X.decompose(mdl=PCA(n_components=2))
        """

        return _data_decomposition(self, mdl)

    def cluster(self, mdl, decomposed=False, pca_comps=4):
        """ Clustering

        Apply one of the following scikit-learn clustering techniques to the stored dataset:

        - KMeans
        - SpectralClustering
        - AgglomerativeClustering

        Parameters
        ----------
        mdl : object
            scikit-learn clustering class

        decomposed : bool
            If true, clusters on the decomposed components

        pca_comps : int
            If `decomposed=True`, this specifies the number of principle components to use.

        Returns
        -------
        lbls : np.ndarray (x, y, (z))
            Returns an image array with the pixels labelled according to the cluster assigned

        spcs : np.ndarray (n_clusters, n_features)
            Returns the spectra of the clusters.

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.cluster import KMeans
        >>> import hypers as hp
        >>> data = np.random.rand(50, 50, 100)
        >>> X = hp.Dataset(data)
        >>> lbls, spcs = X.cluster(mdl=KMeans(n_clusters=3))
        """
        return _data_cluster(self, mdl, decomposed, pca_comps)


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
