import numpy as np
import hypers as hp
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from hypers._tools._types import DecomposeType, DECOMPOSE_TYPES


def _data_decomposition(X: 'hp.Dataset',
                        mdl: DecomposeType,
                        plot: bool,
                        return_arrs: bool) -> Tuple[np.ndarray, np.ndarray]:

    if type(mdl) not in DECOMPOSE_TYPES:
        raise TypeError('Must pass a sklearn decomposition class. Refer to documentation.')

    X.mdl_decompose = mdl
    n_components = X.mdl_decompose.get_params()['n_components']
    images = X.mdl_decompose.fit_transform(X.flatten()).reshape(X.data.shape[:-1] + (n_components,))
    specs = X.mdl_decompose.components_.transpose()

    if plot:
        for component in range(n_components):
            plt.subplot(n_components, 2, 2*component + 1)
            if X.ndim == 3:
                plt.imshow(np.squeeze(images[..., component]))
            elif X.ndim == 4:
                plt.imshow(np.mean(np.squeeze(images[..., component]), -1))
            plt.axis('off')
            plt.subplot(n_components, 2, 2 * component + 2)
            plt.plot(specs[..., component])
        plt.show()

    if return_arrs:
        return images, specs


def _data_scree(X: 'hp.Dataset',
                plot: bool,
                return_arrs: bool) -> np.ndarray:

    mdl = PCA()
    mdl.fit_transform(X.flatten())
    scree = mdl.explained_variance_ratio_

    X._scree = scree

    if plot:
        plt.plot(scree)
        plt.xlabel('Principal components')
        plt.ylabel('Variance ratio')
        plt.title('PCA scree plot')
        plt.show()

    if return_arrs:
        return scree
