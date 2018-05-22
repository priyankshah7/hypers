import pytest
import numpy as np
from sklearn.datasets import make_blobs

from skhyper.process import Process
from skhyper.decomposition import PCA, NMF, FastICA


class TestDecomposition:
    def setup(self):
        data_3d, label_3d = make_blobs(n_samples=64, n_features=32, centers=3)
        data_4d, label_4d = make_blobs(n_samples=128, n_features=32, centers=3)

        data_3d = np.reshape(data_3d, (8, 8, 32))
        data_4d = np.reshape(data_4d, (8, 8, 2, 32))

        self.X_3d = Process(data_3d)
        self.X_4d = Process(data_4d)

    def test_pca(self):
        mdl = PCA()
        with pytest.raises(AttributeError): mdl.plot_statistics()

        mdl_3d = PCA()
        mdl_4d = PCA()

        # 3-dimensional data
        mdl_3d.fit_transform(self.X_3d)
        assert mdl_3d.image_components_ is not None
        assert mdl_3d.spec_components_ is not None
        assert mdl_3d.explained_variance_ is not None
        assert mdl_3d.explained_variance_ratio_ is not None
        assert mdl_3d.singular_values_ is not None
        assert mdl_3d.mean_ is not None
        assert mdl_3d.noise_variance_ is not None

        assert len(mdl_3d.image_components_) == 32
        assert len(mdl_3d.spec_components_) == 32

        Xd_3d = mdl_3d.inverse_transform(n_components=10, perform_anscombe=True)
        assert Xd_3d is not None

        Xd_3d = mdl_3d.inverse_transform(n_components=10, perform_anscombe=False)
        assert Xd_3d is not None

        # 4-dimensional data
        mdl_4d.fit_transform(self.X_4d)
        assert mdl_4d.image_components_ is not None
        assert mdl_4d.spec_components_ is not None
        assert mdl_4d.explained_variance_ is not None
        assert mdl_4d.explained_variance_ratio_ is not None
        assert mdl_4d.singular_values_ is not None
        assert mdl_4d.mean_ is not None
        assert mdl_4d.noise_variance_ is not None

        assert len(mdl_4d.image_components_) == 32
        assert len(mdl_4d.spec_components_) == 32

        Xd_4d = mdl_4d.inverse_transform(n_components=10, perform_anscombe=True)
        assert Xd_4d is not None

        Xd_4d = mdl_4d.inverse_transform(n_components=10, perform_anscombe=False)
        assert Xd_4d is not None

    def test_nmf(self):
        # Passing only positive values
        X_3d = self.X_3d
        X_4d = self.X_4d
        self.X_3d.data = np.abs(X_3d.data)
        self.X_4d.data = np.abs(X_4d.data)
        self.X_3d.update()
        self.X_4d.update()

        mdl_3d = NMF(n_components=3)
        mdl_4d = NMF(n_components=3)

        # 3-dimensional data
        mdl_3d.fit_transform(self.X_3d)
        assert mdl_3d.image_components_ is not None
        assert mdl_3d.spec_components_ is not None
        assert mdl_3d.reconstruction_err_ is not None
        assert mdl_3d.n_iter_ is not None

        assert len(mdl_3d.image_components_) == 3
        assert len(mdl_3d.spec_components_) == 3

        Xd_3d = mdl_3d.inverse_transform(n_components=1)
        assert Xd_3d is not None

        # 4-dimensional data
        mdl_4d.fit_transform(self.X_4d)
        assert mdl_4d.image_components_ is not None
        assert mdl_4d.spec_components_ is not None
        assert mdl_4d.reconstruction_err_ is not None
        assert mdl_4d.n_iter_ is not None

        assert len(mdl_4d.image_components_) == 3
        assert len(mdl_4d.spec_components_) == 3

        Xd_4d = mdl_4d.inverse_transform(n_components=1)
        assert Xd_4d is not None

    def test_fastica(self):
        mdl_3d = FastICA(n_components=3)
        mdl_4d = FastICA(n_components=3)

        # 3-dimensional data
        mdl_3d.fit_transform(self.X_3d)
        assert mdl_3d.image_components_ is not None
        assert mdl_3d.spec_components_ is not None
        assert mdl_3d.mixing_ is not None
        assert mdl_3d.n_iter_ is not None

        assert len(mdl_3d.image_components_) == 3
        assert len(mdl_3d.spec_components_) == 3

        Xd_3d = mdl_3d.inverse_transform(n_components=1)
        assert Xd_3d is not None

        # 4-dimensional data
        mdl_4d.fit_transform(self.X_4d)
        assert mdl_4d.image_components_ is not None
        assert mdl_4d.spec_components_ is not None
        assert mdl_4d.mixing_ is not None
        assert mdl_4d.n_iter_ is not None

        assert len(mdl_4d.image_components_) == 3
        assert len(mdl_4d.spec_components_) == 3

        Xd_4d = mdl_4d.inverse_transform(n_components=1)
        assert Xd_4d is not None
