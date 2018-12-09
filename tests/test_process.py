import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import *
from sklearn.cluster import *

from skhyper.process import Process


class TestProcess:
    def setup(self):
        data_3d, label_3d = make_blobs(n_samples=64, n_features=32, centers=3)
        data_4d, label_4d = make_blobs(n_samples=128, n_features=32, centers=3)

        data_3d = np.abs(np.reshape(data_3d, (8, 8, 32)))
        data_4d = np.abs(np.reshape(data_4d, (8, 8, 2, 32)))

        self.X_3d = Process(data_3d)
        self.X_4d = Process(data_4d)

    def test_decompose(self):
        sk_decompose_types = (
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

        for decomp_type in sk_decompose_types:
            ims_3d, spcs3d = self.X_3d.decompose(mdl=decomp_type(n_components=2))
            ims_4d, spcs4d = self.X_4d.decompose(mdl=decomp_type(n_components=2))

            assert ims_3d.shape == (8, 8, 2)
            assert spcs3d.shape == (32, 2)
            assert ims_4d.shape == (8, 8, 2, 2)
            assert spcs4d.shape == (32, 2)

    def test_cluster(self):
        sk_cluster_types = (
            KMeans,
            # AffinityPropagation,
            # MeanShift,
            SpectralClustering,
            AgglomerativeClustering,
            # DBSCAN,
            # Birch
        )

        for cluster_type in sk_cluster_types:
            print(cluster_type.__name__)
            lbls_3d, spcs3d = self.X_3d.cluster(
                mdl=cluster_type(n_clusters=2), decomposed=False
            )
            lbls_4d, spcs4d = self.X_4d.cluster(
                mdl=cluster_type(n_clusters=2), decomposed=False
            )

            assert lbls_3d.shape == (8, 8)
            assert lbls_4d.shape == (8, 8, 2)
            assert spcs3d.shape == (2, 32)
            assert spcs4d.shape == (2, 32)

            lbls_3d, spcs3d = self.X_3d.cluster(
                mdl=cluster_type(n_clusters=2), decomposed=True, pca_comps=2
            )
            lbls_4d, spcs4d = self.X_4d.cluster(
                mdl=cluster_type(n_clusters=2), decomposed=True, pca_comps=2
            )

            assert lbls_3d.shape == (8, 8)
            assert lbls_4d.shape == (8, 8, 2)
            assert spcs3d.shape == (2, 32)
            assert spcs4d.shape == (2, 32)
