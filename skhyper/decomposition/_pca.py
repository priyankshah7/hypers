import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as _sklearn_pca

from skhyper.utils import HyperanalysisError
from skhyper.process import data_shape, data_tranform2d, data_back_transform
from skhyper.decomposition._anscombe import anscombe_transform, inverse_anscombe_transform
from skhyper.utils._data_checks import _data_checks
from skhyper.utils._plot import _plot_decomposition


# TODO Need to change HyperanalysisError name
class PCA:
    def __init__(self, n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0,
                 iterated_power='auto', random_state=None):
        self.data = None
        self._shape = None
        self._dimensions = None

        # decomposition outputs
        self.data_denoised = None
        self.images = None
        self.spectra = None

        # sklearn PCA model
        self.mdl = None

        # sklearn PCA outputs
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
        if self.data is None:
            raise HyperanalysisError('Data has not yet been fitted with fit()')

    def plot_statistics(self):
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
        self._check_is_fitted()
        _plot_decomposition(plot_range=plot_range, images=self.images, spectra=self.spectra, dim=self._dimensions)

    def fit(self, data):
        self.data = data
        self._shape, self._dimensions = _data_checks(self.data)

        data2d = data_tranform2d(self.data)

        pca_model = _sklearn_pca(n_components=self.n_components, copy=self.copy, whiten=self.whiten,
                                 svd_solver=self.svd_solver, tol=self.tol, iterated_power=self.iterated_power,
                                 random_state=self.random_state)
        w_matrix = pca_model.fit_transform(data2d)
        h_matrix = pca_model.components_

        self.mdl = pca_model
        self.explained_variance_ = pca_model.explained_variance_
        self.explained_variance_ratio_ = pca_model.explained_variance_ratio_
        self.singular_values_ = pca_model.singular_values_
        self.mean_ = pca_model.mean_
        self.noise_variance_ = pca_model.noise_variance_

        self.images = np.reshape(w_matrix, self._shape)
        self.spectra = h_matrix.T

    # TODO Need to figure out what's going in when performing anscombe transformation
    def inverse_transform(self, n_components, perform_anscombe=True):
        self._check_is_fitted()

        shape, dimensions = data_shape(self.data)

        data = self.data
        if perform_anscombe:
            data = anscombe_transform(self.data, gauss_std=0, gauss_mean=0, poisson_multi=1)

        data2d = data_tranform2d(data)

        pca_model = _sklearn_pca(n_components=n_components, copy=self.copy, whiten=self.whiten,
                                 svd_solver=self.svd_solver, tol=self.tol, iterated_power=self.iterated_power,
                                 random_state=self.random_state)
        data_denoised2d = pca_model.fit_transform(data2d)
        data_denoised2d = pca_model.inverse_transform(data_denoised2d)

        data_denoised = data_back_transform(data_denoised2d, shape, dimensions)
        if perform_anscombe:
            data_denoised = inverse_anscombe_transform(data_denoised, gauss_std=0, gauss_mean=0, poisson_multi=1)

        self.data_denoised = data_denoised

    def get_covariance(self):
        self._check_is_fitted()
        return self.mdl.get_covariance()

    def get_params(self, deep=True):
        self._check_is_fitted()
        return self.mdl.get_params(deep=deep)

    def get_precision(self):
        self._check_is_fitted()
        return self.mdl.get_precision()

    def score(self):
        self._check_is_fitted()
        return self.mdl.score(data_tranform2d(self.data))

    def score_samples(self):
        self._check_is_fitted()
        return np.reshape(self.mdl.score_samples(data_tranform2d(self.data)), self._shape[:-1])
