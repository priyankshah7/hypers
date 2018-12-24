import numpy as np
import hypers as hp
from sklearn.datasets import make_blobs


class TestDataset:
    def setup(self):
        data_3d, _ = make_blobs(n_samples=64, n_features=32, centers=3)
        data_4d, _ = make_blobs(n_samples=128, n_features=32, centers=3)

        self.data_3d = np.abs(np.reshape(data_3d, (8, 8, 32)))
        self.data_4d = np.abs(np.reshape(data_4d, (8, 8, 2, 32)))

    def test_smoothing(self):
        X_3d = hp.Dataset(self.data_3d)
        X_3d.smoothing = 'savitzky_golay'
        X_3d.smoothen(window_length=5, polyorder=3)

        X_3d = hp.Dataset(self.data_3d)
        X_3d.smoothing = 'gaussian_filter'
        X_3d.smoothen(sigma=0.5)

        X_4d = hp.Dataset(self.data_4d)
        X_4d.smoothing = 'savitzky_golay'
        X_4d.smoothen(window_length=5, polyorder=3)

        X_4d = hp.Dataset(self.data_4d)
        X_4d.smoothing = 'gaussian_filter'
        X_4d.smoothen(sigma=0.5)

    def test_flatten(self):
        X_3d = hp.Dataset(self.data_3d)
        X_4d = hp.Dataset(self.data_4d)
        flattened_3d = X_3d.flatten()
        flattened_4d = X_4d.flatten()

        assert flattened_3d.shape == (X_3d.shape[0]*X_3d.shape[1], X_3d.shape[2])
        assert flattened_4d.shape == (X_4d.shape[0]*X_4d.shape[1]*X_4d.shape[2], X_4d.shape[3])

    def test_scree(self):
        X_3d = hp.Dataset(self.data_3d)
        X_4d = hp.Dataset(self.data_4d)
        scree_3d = X_3d.scree()
        scree_4d = X_4d.scree()

        assert scree_3d.shape == (X_3d.shape[-1], )
        assert scree_4d.shape == (X_4d.shape[-1], )

    def test_data_access(self):
        X_3d = hp.Dataset(self.data_3d)
        X_4d = hp.Dataset(self.data_4d)

        arr3d = X_3d[:2, :2, :]
        arr4d = X_4d[:2, :2, :1, :]

        assert arr3d.shape == (2, 2, 32)
        assert arr4d.shape == (2, 2, 1, 32)

        X_3d[0, 0, :] = np.random.rand(32)
        X_4d[0, 0, 0, :] = np.random.rand(32)

    def test_arithmetic_operators(self):
        X_3d = hp.Dataset(self.data_3d)
        X_4d = hp.Dataset(self.data_4d)

        spectral_array = np.random.rand(32)

        # Multiplication
        X_3d *= 2
        X_3d *= spectral_array
        X_4d *= 2
        X_4d *= spectral_array

        # Division
        X_3d /= 2
        X_3d /= spectral_array
        X_4d /= 2
        X_4d /= spectral_array

        # Addition
        X_3d += 2
        X_3d += spectral_array
        X_4d += 2
        X_4d += spectral_array

        # Subtraction
        X_3d -= 2
        X_3d -= spectral_array
        X_4d -= 2
        X_4d -= spectral_array
