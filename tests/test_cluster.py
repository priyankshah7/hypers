import pytest
import numpy as np

from skhyper.utils import HyperanalysisError
from skhyper.cluster import KMeans, AgglomerativeClustering, SpectralClustering


class TestCluster:
    def test_kmeans(self):
        data_empty_list = []
        data_1d = np.random.rand(1024)
        data_2d = np.random.rand(20, 1024)
        data_3d = np.random.rand(20, 20, 1024)
        data_4d = np.random.rand(20, 20, 5, 1024)

        # ensure error occurs with a list instead of a numpy array
        with pytest.raises(HyperanalysisError):
            mdl = KMeans(2)
            mdl.fit(data_empty_list)

        # ensure error occurs with 1-dimensional array
        with pytest.raises(HyperanalysisError):
            mdl = KMeans(2)
            mdl.fit(data_1d)

        # ensure error occurs with 2-dimensional array
        with pytest.raises(HyperanalysisError):
            mdl = KMeans(2)
            mdl.fit(data_2d)

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
        with pytest.raises(HyperanalysisError):
            mdl_plot.plot()

    def test_agglomerative_clustering(self):
        data_empty_list = []
        data_1d = np.random.rand(1024)
        data_2d = np.random.rand(20, 1024)
        data_3d = np.random.rand(20, 20, 1024)
        data_4d = np.random.rand(20, 20, 5, 1024)

        # ensure error occurs with a list instead of a numpy array
        with pytest.raises(HyperanalysisError):
            mdl = AgglomerativeClustering(2)
            mdl.fit(data_empty_list)

        # ensure error occurs with 1-dimensional array
        with pytest.raises(HyperanalysisError):
            mdl = AgglomerativeClustering(2)
            mdl.fit(data_1d)

        # ensure error occurs with 2-dimensional array
        with pytest.raises(HyperanalysisError):
            mdl = AgglomerativeClustering(2)
            mdl.fit(data_2d)

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
        with pytest.raises(HyperanalysisError):
            mdl_plot.plot()

    def test_spectral_clustering(self):
        data_empty_list = []
        data_1d = np.random.rand(1024)
        data_2d = np.random.rand(20, 1024)
        data_3d = np.random.rand(20, 20, 1024)
        data_4d = np.random.rand(20, 20, 5, 1024)

        # ensure error occurs with a list instead of a numpy array
        with pytest.raises(HyperanalysisError):
            mdl = SpectralClustering(2)
            mdl.fit(data_empty_list)

        # ensure error occurs with 1-dimensional array
        with pytest.raises(HyperanalysisError):
            mdl = SpectralClustering(2)
            mdl.fit(data_1d)

        # ensure error occurs with 2-dimensional array
        with pytest.raises(HyperanalysisError):
            mdl = SpectralClustering(2)
            mdl.fit(data_2d)

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
        with pytest.raises(HyperanalysisError):
            mdl_plot.plot()
