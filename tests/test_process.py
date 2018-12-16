import pytest
import numpy as np
from sklearn.datasets import make_blobs
import hypers as hp
from sklearn.decomposition import (
    PCA, FastICA, IncrementalPCA, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
    FactorAnalysis, NMF, LatentDirichletAllocation
)
from sklearn.cluster import (
    KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering,
    DBSCAN, Birch
)
from sklearn.preprocessing import (
    MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler,
    StandardScaler
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

CLUSTER_TYPES = (
    KMeans,
    # AffinityPropagation,
    # MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    # DBSCAN,
    # Birch
)

PREPROCESSING_TYPES = (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler
)


class TestProcess:
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

    def test_smoothing(self):
        X_3d = hp.Dataset(self.data_3d)
        X_3d.smoothing = 'savitzky_golay'
        X_3d.smoothen(window_length=5, polyorder=3)

        X_3d = hp.Dataset(self.data_3d)
        X_3d.smoothing = 'gaussian_filter'
        X_3d.smoothen(sigma=0.5)

        X_4d = hp.Dataset(self.data_4d)
        X_4d.smoothing = 'savitzky_golay'
        X_4d.smoothen(window_length=5, polyorder=3)

        X_4d = hp.Dataset(self.data_4d)
        X_4d.smoothing = 'gaussian_filter'
        X_4d.smoothen(sigma=0.5)

    def test_flatten(self):
        X_3d = hp.Dataset(self.data_3d)
        X_4d = hp.Dataset(self.data_4d)
        flattened_3d = X_3d.flatten()
        flattened_4d = X_4d.flatten()

        assert flattened_3d.shape == (X_3d.shape[0]*X_3d.shape[1], X_3d.shape[2])
        assert flattened_4d.shape == (X_4d.shape[0]*X_4d.shape[1]*X_4d.shape[2], X_4d.shape[3])

    def test_scree(self):
        X_3d = hp.Dataset(self.data_3d)
        X_4d = hp.Dataset(self.data_4d)
        scree_3d = X_3d.scree()
        scree_4d = X_4d.scree()

        assert scree_3d.shape == (X_3d.shape[-1], )
        assert scree_4d.shape == (X_4d.shape[-1], )

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

    def test_cluster(self):
        X_3d = hp.Dataset(self.data_3d)
        X_4d = hp.Dataset(self.data_4d)

        for cluster_type in CLUSTER_TYPES:
            if not type(cluster_type()) == AffinityPropagation:
                lbls_3d, spcs3d = X_3d.cluster(
                    mdl=cluster_type(n_clusters=2), 
                    decomposed=False
                )
                lbls_4d, spcs4d = X_4d.cluster(
                    mdl=cluster_type(n_clusters=2), 
                    decomposed=False
                )
            
            elif type(cluster_type()) == AffinityPropagation:
                lbls_3d, spcs3d = X_3d.cluster(
                    mdl=cluster_type(), 
                    decomposed=False
                )
                lbls_4d, spcs4d = X_4d.cluster(
                    mdl=cluster_type(), 
                    decomposed=False
                )

            assert lbls_3d.shape == (8, 8)
            assert lbls_4d.shape == (8, 8, 2)
            assert spcs3d.shape == (2, 32)
            assert spcs4d.shape == (2, 32)

            lbls_3d, spcs3d = X_3d.cluster(
                mdl=cluster_type(n_clusters=2), 
                decomposed=True, 
                pca_comps=2
            )
            lbls_4d, spcs4d = X_4d.cluster(
                mdl=cluster_type(n_clusters=2), 
                decomposed=True, 
                pca_comps=2
            )

            assert lbls_3d.shape == (8, 8)
            assert lbls_4d.shape == (8, 8, 2)
            assert spcs3d.shape == (2, 32)
            assert spcs4d.shape == (2, 32)
