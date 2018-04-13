import pytest
import numpy as np

from skhyper.decomposition import PCA


class TestDecomposition:
    def test_pca(self):
        mdl = PCA()
        with pytest.raises(AttributeError): mdl.plot_statistics()
        with pytest.raises(AttributeError): mdl.plot_components()

        data_3d = np.random.rand(40, 40, 1024)
        data_4d = np.random.rand(40, 40, 5, 1024)

        mdl_3d = PCA()
        mdl_4d = PCA()

        # 3-dimensional data
        mdl_3d.fit(data_3d)
        assert mdl_3d.images is not None
        assert mdl_3d.spectra is not None
        assert mdl_3d.explained_variance_ is not None
        assert mdl_3d.explained_variance_ratio_ is not None
        assert mdl_3d.singular_values_ is not None
        assert mdl_3d.mean_ is not None
        assert mdl_3d.noise_variance_ is not None

        mdl_3d.inverse_transform(10, perform_anscombe=True)
        assert mdl_3d.data_denoised is not None

        mdl_3d.inverse_transform(10, perform_anscombe=False)
        assert mdl_3d.data_denoised is not None

        mdl_3d.get_covariance()
        mdl_3d.get_params()
        mdl_3d.get_precision()
        mdl_3d.score()
        mdl_3d.score_samples()

        # 4-dimensional data
        mdl_4d.fit(data_4d)
        assert mdl_4d.images is not None
        assert mdl_4d.spectra is not None
        assert mdl_4d.explained_variance_ is not None
        assert mdl_4d.explained_variance_ratio_ is not None
        assert mdl_4d.singular_values_ is not None
        assert mdl_4d.mean_ is not None
        assert mdl_4d.noise_variance_ is not None

        mdl_4d.inverse_transform(10, perform_anscombe=True)
        assert mdl_4d.data_denoised is not None

        mdl_4d.inverse_transform(10, perform_anscombe=False)
        assert mdl_4d.data_denoised is not None

        mdl_4d.get_covariance()
        mdl_4d.get_params()
        mdl_4d.get_precision()
        mdl_4d.score()
        mdl_4d.score_samples()

        # test to ensure that number of samples must be greater than number of features
        data_features = np.random.rand(20, 20, 1024)  # 20*20 < 1024
        mdl_features = PCA()

        with pytest.raises(TypeError): mdl_features.fit(data_features)
