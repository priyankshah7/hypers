import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA as _sklearn_fastica

from skhyper.utils import HyperanalysisError
from skhyper.process import data_shape, data_tranform2d, data_back_transform
from skhyper.decomposition._anscombe import anscombe_transform, inverse_anscombe_transform
from skhyper.utils._data_checks import _data_checks
from skhyper.utils._plot import _plot_decomposition


class FastICA:
    def __init__(self, n_components=None, algorithm='parallel', whiten=True, fun='logcosh',
                 fun_args=None, max_iter=200, tol=1e-4, w_init=None, random_state=None):
        self.data = None
        self._shape = None
        self._dimensions = None

        # decomposition outputs
        self.images = None
        self.spectra = None

        # sklearn optional FastICA arguments
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.random_state = random_state

    def fit(self, data):
        self.data = data
        self._shape, self._dimensions = _data_checks(self.data)

        data2d = data_tranform2d(self.data)

        fastica_model = _sklearn_fastica(n_components=self.n_components, algorithm=self.algorithm, whiten=self.whiten,
                                         fun=self.fun, fun_args=self.fun_args, max_iter=self.max_iter, tol=self.tol,
                                         w_init=self.w_init, random_state=self.random_state)
        w_matrix = fastica_model.fit_transform(data2d)
        h_matrix = fastica_model.components_

        self.images = np.reshape(w_matrix, self._shape)
        self.spectra = h_matrix.T

    def plot_components(self, plot_range=(0, 2)):
        if self.data is None:
            raise HyperanalysisError('fit() must be caussed first prior to viewing a plot of the components.')

        _plot_decomposition(plot_range=plot_range, images=self.images, spectra=self.spectra, dim=self._dimensions)

