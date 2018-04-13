import numpy as np
from sklearn.cluster import KMeans as _sklearn_kmeans

from skhyper.process import data_tranform2d
from skhyper.utils._data_checks import _data_checks
from skhyper.utils._plot import _plot_cluster


class KMeans:
    def __init__(self, n_clusters, init='k-means++', n_init=10, max_iter=300,
                 tol=1e-4, precompute_distances='auto', verbose=0, random_state=None,
                 copy_x=True, n_jobs=-1, algorithm='auto'):
        self.data = None
        self._shape = None
        self._dimensions = None

        # clustering outputs
        self.labels = None
        self.data_clusters = None

        # sklearn optional KMeans arguments
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.algorithm = algorithm

    def fit(self, data):
        self.data = data
        self._shape, self._dimensions = _data_checks(self.data)

        data2d = data_tranform2d(self.data)
        kmeans_model = _sklearn_kmeans(n_clusters=self.n_clusters, init=self.init, n_init=self.n_init,
                                       max_iter=self.max_iter, tol=self.tol, precompute_distances=self.precompute_distances,
                                       verbose=self.verbose, random_state=self.random_state, copy_x=self.copy_x,
                                       n_jobs=self.n_jobs, algorithm=self.algorithm).fit(data2d)

        if self._dimensions == 3:
            labels = np.reshape(kmeans_model.labels_, (self._shape[0], self._shape[1]))
            labels += 1
            data_clusters = [0]*self.n_clusters
            for cluster in range(self.n_clusters):
                data_clusters[cluster] = np.zeros((self._shape[0], self._shape[1], self._shape[2]))
                for spectral_pixel in range(self._shape[2]):
                    data_clusters[cluster][:, :, spectral_pixel] = np.multiply(
                        self.data[:, :, spectral_pixel], np.where(labels == cluster + 1, labels, 0) / (cluster + 1))

        elif self._dimensions == 4:
            labels = np.reshape(kmeans_model.labels_, (self._shape[0], self._shape[1], self._shape[2]))
            labels += 1
            data_clusters = [0]*self.n_clusters
            for cluster in range(self.n_clusters):
                data_clusters[cluster] = np.zeros((self._shape[0], self._shape[1], self._shape[2], self._shape[3]))
                for spectral_pixel in range(self._shape[3]):
                    data_clusters[cluster][:, :, :, spectral_pixel] = np.multiply(
                        self.data[:, :, :, spectral_pixel], np.where(labels == cluster + 1, labels, 0) / (cluster + 1))

        else:
            raise TypeError('Error in the size of the data array.')

        self.labels = labels
        self.data_clusters = data_clusters

    def plot(self):
        if self.data is None:
            raise AttributeError('fit() must be called first prior to called plot().')

        if self._dimensions == 3:
            _plot_cluster(self.data_clusters, 3)

        elif self._dimensions == 4:
            _plot_cluster(self.data_clusters, 4)

        else:
            raise AttributeError('Data must be fitted with fit() prior to calling plot()')
