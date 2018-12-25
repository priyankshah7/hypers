import numpy as np
import hypers as hp
from typing import Tuple
from sklearn.decomposition import PCA
from hypers._tools._types import DecomposeType, DECOMPOSE_TYPES


def _data_decomposition(X: hp.Dataset,
                        mdl: DecomposeType,
                        return_arrs: bool) -> Tuple[np.ndarray, np.ndarray]:

    if type(mdl) not in DECOMPOSE_TYPES:
        raise TypeError('Must pass a sklearn decomposition class. Refer to documentation.')

    X.mdl_decompose = mdl
    n_components = X.mdl_decompose.get_params()['n_components']
    images = X.mdl_decompose.fit_transform(X.flatten()).reshape(X.data.shape[:-1] + (n_components,))
    specs = X.mdl_decompose.components_.transpose()

    X._decompose_ims = images
    X._decompose_spcs = specs

    if return_arrs:
        return images, specs


def _data_scree(X: hp.Dataset) -> np.ndarray:

    mdl = PCA()
    mdl.fit_transform(X.flatten())
    scree = mdl.explained_variance_ratio_

    X._scree = scree

    return scree
