"""
Principal component analysis
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as _sklearn_pca

from skhyper.process import data_shape, data_tranform2d, data_back_transform
from skhyper.decomposition._anscombe import anscombe_transform, inverse_anscombe_transform
from skhyper.utils._data_checks import _data_checks, _check_features_samples
from skhyper.utils._plot import _plot_decomposition


class PCA:
    """
    Principal component analysis (PCA)
    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.

    Parameters
    ----------
    n_components :  int, float, None or string
                    Number of components to keep.

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
    X_denoised_ : array, shape (n_samples, n_features)
                  Returns denoised data when inverse_transform() is called

    X_image_components_ : array

    X_spec_components_ : array

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
    """
    def __init__(self, n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0,
                 iterated_power='auto', random_state=None):
        self.X = None
        self._shape = None
        self._dimensions = None

        # sklearn PCA model
        self.mdl = None

        # custom decomposition attributes
        self.X_denoised_ = None
        self.X_image_components_ = None
        self.X_spec_components_ = None

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
        Performs a check to see if self.data is empty. If it is, then fit() has not been called yet.
        """
        if self.X is None:
            raise AttributeError('Data has not yet been fitted with fit()')

    def plot_statistics(self):
        """
        Statistics figures
        Produces 2 matplotlib figures.

        Figure 1 : Explained variance, explained variance ratio, singular values and empirical mean
        Figure 2 : Estimated covariance, estimated precision
        """
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

        plt.figure(facecolor='white')
        plt.subplot(1, 2, 1)
        im = plt.imshow(self.mdl.get_covariance())
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title('Estimated covariance')
        plt.subplot(1, 2, 2)
        im = plt.imshow(self.mdl.get_precision())
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title('Estimated precision')
        plt.tight_layout()
        plt.show()

    def plot_components(self, plot_range=(0, 2)):
        """
        Plots the image/spectral principal components within the selected component range

        Parameters
        ----------
        plot_range : tuple
                     must be between 0 and max(n_features)
        """
        self._check_is_fitted()
        if not isinstance(plot_range, tuple):
            raise TypeError('plot_range must be a tuple with 2 elements specifying min and max components')

        if not len(plot_range) == 2:
            raise TypeError('plot_range must be a tuple with 2 elements specifying min and max components')

        try:
            assert plot_range[0] >= 0
            assert plot_range[0] < self._shape[-1]
            assert plot_range[1] > 0
            assert plot_range[1] < self._shape[-1]
        except:
            raise ValueError('plot_range values must be between 0 and n_features')
        else:
            _plot_decomposition(plot_range=plot_range, images=self.X_image_components_, spectra=self.X_spec_components_, dim=self._dimensions)

    # NOTE This will not work if n_samples < n_features (i.e. is x*y[*z] < spectral_points)
    def fit(self, data):
        """
        Fits the model with data

        Parameters
        ----------
        data : array-like, shape (x, y, spectrum), or (x, y, z, spectrum)
               Training data to fit model to.

        Returns
        -------
        self : object
               Returns the instance itself
        """
        self.X = data
        self._fit()
        return self

    def _fit(self):
        """
        Fits the model with data
        """
        self._shape, self._dimensions = _data_checks(self.X)
        _check_features_samples(self.X)

        X_2d = data_tranform2d(self.X)

        pca_model = _sklearn_pca(n_components=self.n_components, copy=self.copy, whiten=self.whiten,
                                 svd_solver=self.svd_solver, tol=self.tol, iterated_power=self.iterated_power,
                                 random_state=self.random_state)
        w_matrix = pca_model.fit_transform(X_2d)
        h_matrix = pca_model.components_

        self.mdl = pca_model
        self.explained_variance_ = pca_model.explained_variance_
        self.explained_variance_ratio_ = pca_model.explained_variance_ratio_
        self.singular_values_ = pca_model.singular_values_
        self.mean_ = pca_model.mean_
        self.noise_variance_ = pca_model.noise_variance_

        self.X_image_components_ = np.reshape(w_matrix, self._shape)
        self.X_spec_components_ = h_matrix.T

    # TODO Need to figure out what's going in when performing anscombe transformation
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
        """
        self._check_is_fitted()

        shape, dimensions = data_shape(self.X)

        data = self.X
        if perform_anscombe:
            data = anscombe_transform(self.X, gauss_std=gauss_std, gauss_mean=gauss_mean,
                                      poisson_multi=poisson_multi)

        data2d = data_tranform2d(data)

        pca_model = _sklearn_pca(n_components=n_components, copy=self.copy, whiten=self.whiten,
                                 svd_solver=self.svd_solver, tol=self.tol, iterated_power=self.iterated_power,
                                 random_state=self.random_state)
        data_denoised2d = pca_model.fit_transform(data2d)
        data_denoised2d = pca_model.inverse_transform(data_denoised2d)

        data_denoised = data_back_transform(data_denoised2d, shape, dimensions)
        if perform_anscombe:
            data_denoised = inverse_anscombe_transform(data_denoised, gauss_std=0, gauss_mean=0, poisson_multi=1)

        self.X_denoised_ = data_denoised

    def get_covariance(self):
        """
        Compute data covariance with the generative model.

        Returns
        -------
        _covariance : array-like, shape=(n_features, n_features)
                      Estimated covariance of data.
        """
        self._check_is_fitted()
        return self.mdl.get_covariance()

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
               If True, will return the parameters for this estimator and contained
               subobjects that are estimators.

        Returns
        -------
        _params : mapping of string to any
                  Parameter names mapped to their values.
        """
        self._check_is_fitted()
        return self.mdl.get_params(deep=deep)

    def get_precision(self):
        """
        Compute data precision matrix with the generative model.

        Returns
        -------
        _precision : array, shape=(n_features, n_features)
                     Estimated precision of data.
        """
        self._check_is_fitted()
        return self.mdl.get_precision()

    def score(self):
        """
        Return the average log-likelihood of all samples.

        Returns
        -------
        _score : float
                 Average log-likelihood of the samples under the current model
        """
        self._check_is_fitted()
        return self.mdl.score(data_tranform2d(self.X))

    def score_samples(self):
        """
        Return the log-likelihood of each sample.

        Returns
        -------
        _score_samples : array, shape (n_samples,)
                         Log-likelihood of each sample under the current model
        """
        self._check_is_fitted()
        return np.reshape(self.mdl.score_samples(data_tranform2d(self.X)), self._shape[:-1])
