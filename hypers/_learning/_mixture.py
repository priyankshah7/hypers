import numpy as np
import hypers as hp
from typing import Tuple
import matplotlib.pyplot as plt
from hypers._tools._types import MixtureType, MIXTURE_TYPES


def _data_mixture(X: 'hp.Dataset',
                  mdl: MixtureType,
                  plot: bool,
                  return_arrs: bool) -> Tuple[np.ndarray, np.ndarray]:

    if type(mdl) not in MIXTURE_TYPES:
        raise TypeError('Must pass a sklearn mixture class. Refer to documentation.')

    X.mdl_mixture = mdl
    n_components = X.mdl_mixture.get_params()['n_components']
    lbls = X.mdl_mixture.fit_predict(X.flatten()).reshape(X.shape[:-1])
    spectra = X.mdl_mixture.means_.T

    if plot:
        plt.subplot(121)
        if X.ndim == 3:
            plt.imshow(lbls)
        elif X.ndim == 4:
            plt.imshow(np.mean(np.squeeze(lbls), -1))

        plt.subplot(122)
        for component in range(n_components):
            plt.plot(spectra[:, component])
        plt.show()

    if return_arrs:
        return lbls, spectra
