"""
Principal component analysis
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as _sklearn_pca

from skhyper.process import Process
from skhyper.decomposition._anscombe import anscombe_transform, inverse_anscombe_transform
from skhyper.decomposition._plot import _plot_components

sns.set()


class PCA:
    """Principal component analysis (PCA)

    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.

    Parameters
    ----------
    n_components :  int or None
        Number of components to use. If none is passed, all are used.

    copy : bool (default True)
        If False, data passed to fit are overwritten and running fit(X).transform(X)
        will not yield the expected results, use fit_transform(X) instead.

    whiten : bool, optional (default False)
        When True (False by default) the components_ vectors are multiplied by the
        square root of n_samples and then divided by the singular values to
        ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal (the
        relative variance scales of the components) but can sometime improve the
        predictive accuracy of the downstream estimators by making their data respect
        some hard-wired assumptions.

    svd_solver : string {‘auto’, ‘full’, ‘arpack’, ‘randomized’}
        auto :
            the solver is selected by a default policy based on X.shape and
            n_components: if the input data is larger than 500x500 and the number of components
            to extract is lower than 80% of the smallest dimension of the data, then the
            more efficient ‘randomized’ method is enabled. Otherwise the exact full SVD is
            computed and optionally truncated afterwards.
        full :
            run exact full SVD calling the standard LAPACK solver via scipy.linalg.svd and
            select the components by postprocessing
        arpack :
            run SVD truncated to n_components calling ARPACK solver via scipy.sparse.linalg.svds.
            It requires strictly 0 < n_components < X.shape[1]
        randomized :
            run randomized SVD

    tol : float >= 0, optional (default .0)
        Tolerance for singular values computed by svd_solver == ‘arpack’.

    iterated_power : int >= 0, or ‘auto’, (default ‘auto’)
        Number of iterations for the power method computed by svd_solver == ‘randomized’.

    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by
        np.random. Used when svd_solver == ‘arpack’ or ‘randomized’.

    Attributes
    ----------
    image_components_ : list
        Returns a list of image arrays for each component. The size is:
            - n_components: If the number of components were specified
            - n_features: If the number of components were not specified

    spec_components_ : list
        Returns a list of spectral arrays for each component. The size is:
            - n_components: If the number of components were specified
            - n_features: If the number of components were not specified

    mdl : object, PCA() instance
        An instance of the PCA model from scikit-learn used when instantiating PCA() from scikit-hyper

    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components. Equal
        to n_components largest eigenvalues of the covariance matrix of X.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and
        the sum of explained variances is equal to 1.0.

    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components. The
        singular values are equal to the 2-norms of the ``n_components`` variables
        in the lower-dimensional space.

    mean_ : array, shape (n_features,)
        Per-feature empirical mean, estimated from the training set. Equal to `X.mean(axis=0)`.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model from
        Tipping and Bishop 1999. See "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574 or http://www.miketipping.com/papers/met-mppca.pdf.
        It is required to computed the estimated data covariance and score samples. Equal
        to the average of (min(n_features, n_samples) - n_components) smallest
        eigenvalues of the covariance matrix of X.


    Examples
    --------
    >>> import numpy as np
    >>> from skhyper.process import Process
    >>> from skhyper.decomposition import PCA
    >>>
    >>> test_data = np.random.rand(100, 100, 10, 1024)
    >>> X = Process(test_data)
    >>>
    >>> mdl = PCA()
    >>> mdl.fit_transform(X)
    >>> Xd = mdl.inverse_transform(n_components=100, perform_anscombe=False)
    >>>
    >>> # X and Xd have the same shape, however Xd only retains the first 100 principal
    >>> # components (and thus, is 'denoised'). Use the skree plot to help you determine
    >>> # how many components to keep.
    """
    def __init__(self, n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0,
                 iterated_power='auto', random_state=None):
        self._X = None

        # sklearn PCA model
        self.mdl = None

        # custom decomposition attributes
        self.image_components_ = None
        self.spec_components_ = None

        # sklearn PCA attributes
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.noise_variance_ = None

        # sklearn optional PCA arguments
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def _check_is_fitted(self):
        """
        Performs a check to see if self.data is empty. If it is, then fit_transform() has not been called yet.
        """
        if self._X is None:
            raise AttributeError('Data has not yet been fitted with fit_transform()')

    def plot(self, plt_type):
        # TODO add number of components here
        if plt_type == 'components':
            title = 'PCA'
            _plot_components(self.image_components_, self.spec_components_, title)

        elif plt_type == 'statistics':
            # TODO remove plot_statistics()
            self._check_is_fitted()

            plt.figure(facecolor='white')
            plt.subplot(2, 2, 1)
            plt.plot(self.explained_variance_)
            plt.xlabel('Principle compoenent no.')
            plt.ylabel('Variance')
            plt.title('Explained variance')
            plt.subplot(2, 2, 2)
            plt.plot(self.explained_variance_ratio_)
            plt.xlabel('Principle compoenent no.')
            plt.ylabel('Variance ratio')
            plt.title('Explained variance ratio')
            plt.subplot(2, 2, 3)
            plt.plot(self.singular_values_)
            plt.xlabel('Principle compoenent no.')
            plt.ylabel('Singular values')
            plt.title('Singular values')
            plt.subplot(2, 2, 4)
            plt.plot(self.mean_)
            plt.xlabel('Spectrum')
            plt.ylabel('Intensity')
            plt.title('Empirical mean')
            plt.tight_layout()
            plt.show()

    def fit_transform(self, X):
        """
        Fits the PCA model to the processed hyperspectral array.

        Parameters
        ----------
        X : object, type (Process)

        Returns
        -------
        self : object
            Returns the instance itself

        """
        self._X = X
        if not isinstance(self._X, Process):
            raise TypeError('Data needs to be passed to skhyper.process.Process first')

        mdl = _sklearn_pca(n_components=self.n_components, copy=self.copy, whiten=self.whiten,
                           svd_solver=self.svd_solver, tol=self.tol, iterated_power=self.iterated_power,
                           random_state=self.random_state)

        w_matrix = mdl.fit_transform(self._X.flatten())
        h_matrix = mdl.components_

        self.mdl = mdl
        self.explained_variance_ = mdl.explained_variance_
        self.explained_variance_ratio_ = mdl.explained_variance_ratio_
        self.singular_values_ = mdl.singular_values_
        self.mean_ = mdl.mean_
        self.noise_variance_ = mdl.noise_variance_

        if not self.n_components:
            self.image_components_ = [0] * self._X.shape[-1]
            self.spec_components_ = [0] * self._X.shape[-1]

            w_matrix_ord = np.reshape(w_matrix, self._X.shape)
            h_matrix_ord = h_matrix.T

            for comp in range(self._X.shape[-1]):
                self.image_components_[comp] = np.squeeze(w_matrix_ord[..., comp])
                self.spec_components_[comp] = h_matrix_ord[..., comp]

        else:
            self.image_components_ = [0] * self.n_components
            self.spec_components_ = [0] * self.n_components

            w_matrix_ord = np.reshape(w_matrix, self._X.shape[:-1] + (self.n_components,))
            h_matrix_ord = h_matrix.T

            for comp in range(self.n_components):
                self.image_components_[comp] = np.squeeze(w_matrix_ord[..., comp])
                self.spec_components_[comp] = h_matrix_ord[..., comp]

        return self

    def inverse_transform(self, n_components, perform_anscombe=False, gauss_std=0, gauss_mean=0,
                          poisson_multi=1):
        """
        Performs an inverse transform on the fitted data to project back to the original space.

        Parameters
        ----------
        n_components : int
            Number of components to keep when projected back to original space

        perform_anscombe : bool, optional (default False)
            Choose whether to perform Anscombe transformation prior to projecting

        gauss_std : int, optional (default 0)
            Standard deviation of data, for Anscombe transformation

        gauss_mean : int, optional (default 0)
            Mean of data, for Anscombe transformation

        poisson_multi : int, optional (default 1)
            Poisson multiplier, for Anscombe transformation

        Returns
        -------
        Xd : object, Process instance of Xd
             Denoised hyperspectral data with the same dimensions as the array passed to fit_transform()

        """
        self._check_is_fitted()

        X_flat = self._X.flatten()
        if perform_anscombe:
            X_flat = anscombe_transform(X_flat, gauss_std=gauss_std, gauss_mean=gauss_mean,
                                        poisson_multi=poisson_multi)

        mdl = _sklearn_pca(n_components=n_components, copy=self.copy, whiten=self.whiten,
                           svd_solver=self.svd_solver, tol=self.tol, iterated_power=self.iterated_power,
                           random_state=self.random_state)

        Xd_flat = mdl.fit_transform(X_flat)
        Xd_flat = mdl.inverse_transform(Xd_flat)

        if perform_anscombe:
            Xd_flat = inverse_anscombe_transform(Xd_flat, gauss_std=gauss_std, gauss_mean=gauss_mean,
                                                 poisson_multi=poisson_multi)

        Xd = np.reshape(Xd_flat, self._X.shape)

        return Process(Xd)
