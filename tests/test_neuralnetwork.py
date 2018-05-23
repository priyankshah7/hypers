import numpy as np
from sklearn.datasets import make_blobs

from skhyper.process import Process
from skhyper.neural_network import MLPClassifier


class TestNeuralNetwork:
    def setup(self):
        data_3d, label_3d = make_blobs(n_samples=64, n_features=16, centers=3)
        data_4d, label_4d = make_blobs(n_samples=128, n_features=16, centers=3)

        data_3d = np.reshape(data_3d, (8, 8, 16))
        label_3d = np.reshape(label_3d, (8, 8))
        data_4d = np.reshape(data_4d, (8, 8, 2, 16))
        label_4d = np.reshape(label_4d, (8, 8, 2))

        self.label_3d_train = label_3d[:, :6]
        self.label_3d_test = label_3d[:, 6:]
        self.X_3d_train = Process(data_3d[:, :6, :])
        self.X_3d_test = Process(data_3d[:, 6:, :])

        self.label_4d_train = label_4d[:, :6, :]
        self.label_4d_test = label_4d[:, 6:, :]
        self.X_4d_train = Process(data_4d[:, :6, :, :])
        self.X_4d_test = Process(data_4d[:, 6:, :, :])

    def test_mlp_classifier(self):
        mdl_3d = MLPClassifier()
        mdl_4d = MLPClassifier()

        # 3-dimensional data
        mdl_3d.fit(self.X_3d_train, self.label_3d_train)

        y_pred_3d = mdl_3d.predict(self.X_3d_test)
        y_pred_prob_3d = mdl_3d.predict_proba(self.X_3d_test)
        score = mdl_3d.score(self.X_3d_test, self.label_3d_test)

        assert y_pred_3d is not None
        assert y_pred_prob_3d is not None
        assert score is not None

        # 4-dimensional data
        mdl_4d.fit(self.X_4d_train, self.label_4d_train)

        y_pred_4d = mdl_4d.predict(self.X_4d_test)
        y_pred_prob_4d = mdl_4d.predict_proba(self.X_4d_test)
        score = mdl_4d.score(self.X_4d_test, self.label_4d_test)

        assert y_pred_4d is not None
        assert y_pred_prob_4d is not None
        assert score is not None
