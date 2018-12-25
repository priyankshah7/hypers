import numpy as np
import hypers as hp
from hypers._tools._types import PreprocessType, PREPROCESSING_TYPES


def _data_preprocessing(X: hp.Dataset,
                        mdl: PreprocessType) -> None:

    if type(mdl) not in PREPROCESSING_TYPES:
        raise TypeError('Must pass a sklearn preprocessing class. Refer to documentation.')

    X.mdl_preprocess = mdl

    X_newdata = X.mdl_preprocess.fit_transform(X.flatten()).reshape(X.shape)
    X.data = X_newdata
    X.update()


def _data_scale(X: hp.Dataset) -> None:
    if X.ndim == 3:
        for _x in range(X.shape[0]):
            for _y in range(X.shape[1]):
                X.data[_x, _y, :] /= np.max(X.data[_x, _y, :])

    if X.ndim == 4:
        for _x in range(X.shape[0]):
            for _y in range(X.shape[1]):
                for _z in range(X.shape[2]):
                    X.data[_x, _y, _z, :] /= np.max(X.data[_x, _y, _z, :])

    X.update()
