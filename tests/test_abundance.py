import numpy as np
import hypers as hp
from sklearn.datasets import make_blobs


class TestAbundance:
    def setup(self):
        data_3d, _ = make_blobs(n_samples=64, n_features=32, centers=3)
        data_4d, _ = make_blobs(n_samples=128, n_features=32, centers=3)

        self.data_3d = np.abs(np.reshape(data_3d, (8, 8, 32)))
        self.data_4d = np.abs(np.reshape(data_4d, (8, 8, 2, 32)))

    def test_abundance(self):
        X_3d = hp.Dataset(self.data_3d)
        X_4d = hp.Dataset(self.data_4d)

        lbls_3d = X_3d.abundance(
            spectra=np.random.rand(32, 5),
            plot=False,
            return_arrs=True
        )

        lbls_4d = X_4d.abundance(
            spectra=np.random.rand(32, 5),
            plot=False,
            return_arrs=True
        )

        assert lbls_3d.shape == (8, 8, 5)
        assert lbls_4d.shape == (8, 8, 2, 5)
