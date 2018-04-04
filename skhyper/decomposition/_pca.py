import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as _sklearn_pca

from skhyper.utils import HyperanalysisError
from skhyper.process import data_shape, data_tranform2d, data_back_transform
from skhyper.decomposition._anscombe import anscombe_transform, inverse_anscombe_transform
from skhyper.utils._data_checks import _data_checks
from skhyper.utils._plot import _plot_decomposition


# TODO Add the other PCA methods and attributes
class PCA:
    def __init__(self, n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0,
                 iterated_power='auto', random_state=None):
        self.data = None
        self._shape = None
        self._dimensions = None

        # decomposition outputs
        self.data_variance = None
        self.data_denoised = None
        self.images = None
        self.spectra = None

        # sklearn optional PCA arguments
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def fit(self, data):
        self.data = data
        self._shape, self._dimensions = _data_checks(self.data)

        data2d = data_tranform2d(self.data)

        pca_model = _sklearn_pca(n_components=self.n_components, copy=self.copy, whiten=self.whiten,
                                 svd_solver=self.svd_solver, tol=self.tol, iterated_power=self.iterated_power,
                                 random_state=self.random_state)
        w_matrix = pca_model.fit_transform(data2d)
        h_matrix = pca_model.components_
        data_variance = pca_model.explained_variance_ratio_

        self.images = np.reshape(w_matrix, self._shape)
        self.spectra = h_matrix.T
        self.data_variance = data_variance

    def plot_variance(self):
        if self.data is None:
            raise HyperanalysisError('fit() must be called first prior to viewing the skree plot.')

        plt.figure(facecolor='white')
        plt.plot(self.data_variance)
        plt.xlabel('Principle compoenent no.')
        plt.ylabel('Variance contribution (0-1)')
        plt.title('Variance contribution of each principle component')
        plt.show()

    def plot_components(self, plot_range=(0, 2)):
        if self.data is None:
            raise HyperanalysisError('fit() must be caussed first prior to viewing a plot of the components.')

        _plot_decomposition(plot_range=plot_range, images=self.images, spectra=self.spectra, dim=self._dimensions)

    # TODO Need to figure out what's going in when performing anscombe transformation
    def inverse_transform(self, n_components, perform_anscombe=True):
        if self.data is None:
            raise HyperanalysisError('fit() must be called first prior to performing inverse_transform().')

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
