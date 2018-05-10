import numpy as np

from skhyper.process import Process
from skhyper.cluster import KMeans


class TestCluster:
    def test_kmeans(self):
        data_3d = np.random.rand(6, 6, 25)
        data_4d = np.random.rand(5, 5, 2, 25)

        X_3d = Process(data_3d)
        X_4d = Process(data_4d)

        # ensure that 3- and 4-dimensional data produce non-zero labels and data clusters
        mdl_3d = KMeans(n_clusters=2)
        mdl_3d.fit(X_3d)
        assert mdl_3d.labels_ is not None

        mdl_4d = KMeans(n_clusters=2)
        mdl_4d.fit(X_4d)
        assert mdl_4d.labels_ is not None
