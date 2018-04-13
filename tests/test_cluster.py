import pytest
import numpy as np

from skhyper.cluster import KMeans, AgglomerativeClustering, SpectralClustering


class TestCluster:
    def test_kmeans(self):
        data_3d = np.random.rand(20, 20, 1024)
        data_4d = np.random.rand(20, 20, 5, 1024)

        # ensure that 3- and 4-dimensional data produce non-zero labels and data clusters
        mdl_3d = KMeans(2)
        mdl_3d.fit(data_3d)
        assert mdl_3d.labels is not None
        assert mdl_3d.data_clusters is not None

        mdl_4d = KMeans(2)
        mdl_4d.fit(data_4d)
        assert mdl_4d.labels is not None
        assert mdl_4d.data_clusters is not None

        # ensure that plot() cannot be called prior to calling fit()
        mdl_plot = KMeans(2)
        with pytest.raises(AttributeError): mdl_plot.plot()

    def test_agglomerative_clustering(self):
        data_3d = np.random.rand(20, 20, 1024)
        data_4d = np.random.rand(20, 20, 5, 1024)

        # ensure that 3- and 4-dimensional data produce non-zero labels and data clusters
        mdl_3d = AgglomerativeClustering(2)
        mdl_3d.fit(data_3d)
        assert mdl_3d.labels is not None
        assert mdl_3d.data_clusters is not None

        mdl_4d = AgglomerativeClustering(2)
        mdl_4d.fit(data_4d)
        assert mdl_4d.labels is not None
        assert mdl_4d.data_clusters is not None

        # ensure that plot() cannot be called prior to calling fit()
        mdl_plot = AgglomerativeClustering(2)
        with pytest.raises(AttributeError): mdl_plot.plot()

    def test_spectral_clustering(self):
        data_3d = np.random.rand(20, 20, 1024)
        data_4d = np.random.rand(20, 20, 5, 1024)

        # ensure that 3- and 4-dimensional data produce non-zero labels and data clusters
        mdl_3d = SpectralClustering(2)
        mdl_3d.fit(data_3d)
        assert mdl_3d.labels is not None
        assert mdl_3d.data_clusters is not None

        mdl_4d = SpectralClustering(2)
        mdl_4d.fit(data_4d)
        assert mdl_4d.labels is not None
        assert mdl_4d.data_clusters is not None

        # ensure that plot() cannot be called prior to calling fit()
        mdl_plot = SpectralClustering(2)
        with pytest.raises(AttributeError): mdl_plot.plot()
