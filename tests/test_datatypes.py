import pytest
import numpy as np

from skhyper.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from skhyper.decomposition import PCA


class TestDataTypes:
    def test_data_types(self):
        data_empty_list = []
        data_1d = np.random.rand(1024)
        data_2d = np.random.rand(20, 1024)

        # clustering techniques
        mdl_kmeans = KMeans(2)
        mdl_agglomerative = AgglomerativeClustering(2)
        mdl_spectral = SpectralClustering(2)

        # decomposition techniques
        mdl_pca = PCA()

        # ensure error occurs with a list instead of a numpy array
        with pytest.raises(TypeError): mdl_kmeans.fit(data_empty_list)
        with pytest.raises(TypeError): mdl_agglomerative.fit(data_empty_list)
        with pytest.raises(TypeError): mdl_spectral.fit(data_empty_list)
        with pytest.raises(TypeError): mdl_pca.fit(data_empty_list)

        # ensure error occurs with 1-dimensional array
        with pytest.raises(TypeError): mdl_kmeans.fit(data_1d)
        with pytest.raises(TypeError): mdl_agglomerative.fit(data_1d)
        with pytest.raises(TypeError): mdl_spectral.fit(data_1d)
        with pytest.raises(TypeError): mdl_pca.fit(data_1d)

        # ensure error occurs with 2-dimensional array
        with pytest.raises(TypeError): mdl_kmeans.fit(data_2d)
        with pytest.raises(TypeError): mdl_agglomerative.fit(data_2d)
        with pytest.raises(TypeError): mdl_spectral.fit(data_2d)
        with pytest.raises(TypeError): mdl_pca.fit(data_2d)
