import numpy as np
import hypers as hp
from sklearn.datasets import make_blobs
from sklearn.cluster import (
    KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN
)

CLUSTER_TYPES = (
    KMeans,
    AffinityPropagation,
    MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    DBSCAN
)


class TestCluster:
    def setup(self):
        data_3d, _ = make_blobs(n_samples=64, n_features=32, centers=3)
        data_4d, _ = make_blobs(n_samples=128, n_features=32, centers=3)

        self.data_3d = np.abs(np.reshape(data_3d, (8, 8, 32)))
        self.data_4d = np.abs(np.reshape(data_4d, (8, 8, 2, 32)))

    def test_cluster(self):
        X_3d = hp.Dataset(self.data_3d)
        X_4d = hp.Dataset(self.data_4d)

        for cluster_type in CLUSTER_TYPES:
            if not type(cluster_type()) in (AffinityPropagation, DBSCAN, MeanShift):
                lbls_3d, spcs3d = X_3d.cluster(
                    mdl=cluster_type(n_clusters=2),
                    decomposed=False
                )
                lbls_4d, spcs4d = X_4d.cluster(
                    mdl=cluster_type(n_clusters=2),
                    decomposed=False
                )

                lbls_decomp_3d, spcs_decomp_3d = X_3d.cluster(
                    mdl=cluster_type(n_clusters=2),
                    decomposed=True,
                    pca_comps=2
                )
                lbls_decomp_4d, spcs_decomp_4d = X_4d.cluster(
                    mdl=cluster_type(n_clusters=2),
                    decomposed=True,
                    pca_comps=2
                )

                assert lbls_3d.shape == (8, 8)
                assert lbls_4d.shape == (8, 8, 2)
                assert spcs3d.shape == (32, 2)
                assert spcs4d.shape == (32, 2)
                assert lbls_decomp_3d.shape == (8, 8)
                assert lbls_decomp_4d.shape == (8, 8, 2)
                assert spcs_decomp_3d.shape == (32, 2)
                assert spcs_decomp_4d.shape == (32, 2)

            elif type(cluster_type()) in (AffinityPropagation, DBSCAN, MeanShift):
                X_3d.cluster(
                    mdl=cluster_type(),
                    decomposed=False
                )
                X_4d.cluster(
                    mdl=cluster_type(),
                    decomposed=False
                )

                X_3d.cluster(
                    mdl=cluster_type(),
                    decomposed=True,
                    pca_comps=2
                )
                X_4d.cluster(
                    mdl=cluster_type(),
                    decomposed=True,
                    pca_comps=2
                )
