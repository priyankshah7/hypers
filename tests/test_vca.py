import numpy as np
import hypers as hp
from sklearn.datasets import make_blobs


class TestVCA:
    def setup(self):
        data_3d, _ = make_blobs(n_samples=64, n_features=32, centers=3)
        data_4d, _ = make_blobs(n_samples=128, n_features=32, centers=3)

        self.data_3d = np.abs(np.reshape(data_3d, (8, 8, 32)))
        self.data_4d = np.abs(np.reshape(data_4d, (8, 8, 2, 32)))

    def test_vca(self):
        X_3d = hp.Dataset(self.data_3d)
        X_4d = hp.Dataset(self.data_4d)

        spectra_3d, coords_3d = X_3d.vca(
            n_components=2,
            plot=False,
            return_arrs=True
        )

        spectra_4d, coords_4d = X_4d.vca(
            n_components=2,
            plot=False,
            return_arrs=True
        )

        assert spectra_3d.shape == (32, 2)
        assert spectra_4d.shape == (32, 2)
