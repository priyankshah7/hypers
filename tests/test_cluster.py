import numpy as np
from sklearn.datasets import make_blobs

from skhyper.process import Process
from skhyper.cluster import KMeans


class TestCluster:
    def setup(self):
        data_3d, label_3d = make_blobs(n_samples=64, n_features=32, centers=3)
        data_4d, label_4d = make_blobs(n_samples=128, n_features=32, centers=3)

        data_3d = np.reshape(data_3d, (8, 8, 32))
        data_4d = np.reshape(data_4d, (8, 8, 2, 32))

        self.X_3d = Process(data_3d)
        self.X_4d = Process(data_4d)

    def test_kmeans(self):
        # ensure that 3- and 4-dimensional data produce non-zero labels and data clusters
        mdl_3d = KMeans(n_clusters=2)
        mdl_3d.fit(self.X_3d)
        assert mdl_3d.labels_ is not None
        assert mdl_3d.spec_components_ is not None
        assert mdl_3d.image_components_ is not None
        assert len(mdl_3d.spec_components_) == 2
        assert len(mdl_3d.image_components_) == 2

        mdl_4d = KMeans(n_clusters=2)
        mdl_4d.fit(self.X_4d)
        assert mdl_4d.labels_ is not None
        assert mdl_4d.spec_components_ is not None
        assert mdl_4d.image_components_ is not None
        assert len(mdl_4d.spec_components_) == 2
        assert len(mdl_4d.image_components_) == 2
