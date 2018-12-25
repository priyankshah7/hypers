import numpy as np
import hypers as hp
from sklearn.datasets import make_blobs
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


class TestPreprocess:
    def setup(self):
        data_3d, _ = make_blobs(n_samples=64, n_features=32, centers=3)
        data_4d, _ = make_blobs(n_samples=128, n_features=32, centers=3)

        self.data_3d = np.abs(np.reshape(data_3d, (8, 8, 32)))
        self.data_4d = np.abs(np.reshape(data_4d, (8, 8, 2, 32)))

    def test_preprocessing(self):
        X_3d = hp.Dataset(self.data_3d)
        X_4d = hp.Dataset(self.data_4d)

        for preprocess_type in PREPROCESSING_TYPES:
            X_3d.preprocess(
                mdl=preprocess_type()
            )

            X_4d.preprocess(
                mdl=preprocess_type()
            )
