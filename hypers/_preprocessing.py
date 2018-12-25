import numpy as np
from sklearn.preprocessing import (
    MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler,
    StandardScaler, Normalizer
)

PREPROCESSING_TYPES = (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    Normalizer
)


def _data_preprocessing(X, mdl):
    if type(mdl) not in PREPROCESSING_TYPES:
        raise TypeError('Must pass a sklearn preprocessing class. Refer to documentation.')
    
    X.mdl_preprocess = mdl
    
    X_newdata = X.mdl_preprocess.fit_transform(X.flatten()).reshape(X.shape)
    X.data = X_newdata
    X.update()


def _data_scale(X):
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
