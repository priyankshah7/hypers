import numpy as np
import hypers as hp
from typing import Tuple
from sklearn.mixture import GaussianMixture


class mixture_models:
    """ Provides instance of mixture classes """
    def __init__(self, X: 'hp.hparray'):
        self.gaussian_mixture = gaussian_mixture(X)


class gaussian_mixture:
    def __init__(self, X: 'hp.hparray'):
        self.X = X
        self.labels = None
        self.spcs = None

    def calculate(self, n_components: int=None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if n_components is None:
            n_components = self.X.shape[-1]

        mdl = GaussianMixture(n_components=n_components, **kwargs)
        self.labels = mdl.fit_predict(self.X.collapse()).reshape(self.X.shape[:-1])
        self.spcs = mdl.means_.T

        return self.labels, self.spcs
