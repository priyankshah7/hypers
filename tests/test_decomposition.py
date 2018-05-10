import pytest
import numpy as np

from skhyper.process import Process
from skhyper.decomposition import PCA


class TestDecomposition:
    def test_pca(self):
        mdl = PCA()
        with pytest.raises(AttributeError): mdl.plot_statistics()

        data_3d = np.random.rand(6, 6, 25)
        data_4d = np.random.rand(5, 5, 2, 25)

        X_3d = Process(data_3d)
        X_4d = Process(data_4d)

        mdl_3d = PCA()
        mdl_4d = PCA()

        # 3-dimensional data
        mdl_3d.fit_transform(X_3d)
        assert mdl_3d.image_components_ is not None
        assert mdl_3d.spec_components_ is not None
        assert mdl_3d.explained_variance_ is not None
        assert mdl_3d.explained_variance_ratio_ is not None
        assert mdl_3d.singular_values_ is not None
        assert mdl_3d.mean_ is not None
        assert mdl_3d.noise_variance_ is not None

        Xd_3d = mdl_3d.inverse_transform(n_components=10, perform_anscombe=True)
        assert Xd_3d is not None

        Xd_3d = mdl_3d.inverse_transform(n_components=10, perform_anscombe=False)
        assert Xd_3d is not None

        # 4-dimensional data
        mdl_4d.fit_transform(X_4d)
        assert mdl_4d.image_components_ is not None
        assert mdl_4d.spec_components_ is not None
        assert mdl_4d.explained_variance_ is not None
        assert mdl_4d.explained_variance_ratio_ is not None
        assert mdl_4d.singular_values_ is not None
        assert mdl_4d.mean_ is not None
        assert mdl_4d.noise_variance_ is not None

        Xd_4d = mdl_4d.inverse_transform(n_components=10, perform_anscombe=True)
        assert Xd_4d is not None

        Xd_4d = mdl_4d.inverse_transform(n_components=10, perform_anscombe=False)
        assert Xd_4d is not None

