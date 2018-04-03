import numpy as np
from sklearn.cluster import SpectralClustering as _sklearn_spectral_clustering

from skhyper.utils import HyperanalysisError
from skhyper.process import data_tranform2d
from skhyper.utils._data_checks import _data_checks
from skhyper.utils._plot import _plot_cluster


class SpectralClustering:
    def __init__(self, n_clusters, eigen_solver=None, random_state=None, n_init=10, gamma=1.,
                 affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans',
                 degree=3, coef0=1, kernel_params=None, n_jobs=1):
        self.data = None
        self._shape = None
        self._dimensions = None

        # clustering outputs
        self.labels = None
        self.data_clusters = None

        # sklearn optional SpectralClustering arguments
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs

    def fit(self, data):
        self.data = data
        self._shape, self._dimensions = _data_checks(self.data)

        data2d = data_tranform2d(self.data)
        spectral_model = _sklearn_spectral_clustering(n_clusters=self.n_clusters, eigen_solver=self.eigen_solver,
                                                      random_state=self.random_state, n_init=self.n_init,
                                                      gamma=self.gamma, affinity=self.affinity,
                                                      n_neighbors=self.n_neighbors, eigen_tol=self.eigen_tol,
                                                      assign_labels=self.assign_labels, degree=self.degree,
                                                      coef0=self.coef0, kernel_params=self.kernel_params,
                                                      n_jobs=self.n_jobs).fit(data2d)

        if self._dimensions == 3:
            labels = np.reshape(spectral_model.labels_, (self._shape[0], self._shape[1]))
            labels += 1
            data_clusters = [0] * self.n_clusters
            for cluster in range(self.n_clusters):
                data_clusters[cluster] = np.zeros((self._shape[0], self._shape[1], self._shape[2]))
                for spectral_pixel in range(self._shape[2]):
                    data_clusters[cluster][:, :, spectral_pixel] = np.multiply(
                        self.data[:, :, spectral_pixel], np.where(labels == cluster + 1, labels, 0) / (cluster + 1))

        elif self._dimensions == 4:
            labels = np.reshape(spectral_model.labels_, (self._shape[0], self._shape[1], self._shape[2]))
            labels += 1
            data_clusters = [0] * self.n_clusters
            for cluster in range(self.n_clusters):
                data_clusters[cluster] = np.zeros((self._shape[0], self._shape[1], self._shape[2], self._shape[3]))
                for spectral_pixel in range(self._shape[3]):
                    data_clusters[cluster][:, :, :, spectral_pixel] = np.multiply(
                        self.data[:, :, :, spectral_pixel], np.where(labels == cluster + 1, labels, 0) / (cluster + 1))

        else:
            raise HyperanalysisError('Error in the size of the data array.')

        self.labels = labels
        self.data_clusters = data_clusters

    def plot(self):
        if self.data is None:
            raise HyperanalysisError('fit() must be called first prior to called plot().')

        if self._dimensions == 3:
            _plot_cluster(self.data_clusters, 3)

        elif self._dimensions == 4:
            _plot_cluster(self.data_clusters, 4)

        else:
            raise HyperanalysisError('Data must be fitted with fit() prior to calling plot()')
