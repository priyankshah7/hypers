import numpy as np
import hypers as hp
from sklearn.datasets import make_blobs
from sklearn.decomposition import (
    PCA, FastICA, IncrementalPCA, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
    FactorAnalysis, NMF, LatentDirichletAllocation
)

DECOMPOSE_TYPES = (
    PCA,
    FastICA,
    #KernelPCA,
    IncrementalPCA,
    TruncatedSVD,
    DictionaryLearning,
    MiniBatchDictionaryLearning,
    FactorAnalysis,
    NMF,
    LatentDirichletAllocation
)


class TestDecompose:
    def setup(self):
        data_3d, _ = make_blobs(n_samples=64, n_features=32, centers=3)
        data_4d, _ = make_blobs(n_samples=128, n_features=32, centers=3)

        self.data_3d = np.abs(np.reshape(data_3d, (8, 8, 32)))
        self.data_4d = np.abs(np.reshape(data_4d, (8, 8, 2, 32)))

    def test_decompose(self):
        X_3d = hp.Dataset(self.data_3d)
        X_4d = hp.Dataset(self.data_4d)

        for decomp_type in DECOMPOSE_TYPES:
            ims_3d, spcs3d = X_3d.decompose(
                mdl=decomp_type(n_components=2)
            )
            ims_4d, spcs4d = X_4d.decompose(
                mdl=decomp_type(n_components=2)
            )

            assert ims_3d.shape == (8, 8, 2)
            assert spcs3d.shape == (32, 2)
            assert ims_4d.shape == (8, 8, 2, 2)
            assert spcs4d.shape == (32, 2)
