import numpy as np
from sklearn.cluster import AgglomerativeClustering as _sklearn_agglomerative_clustering

from skhyper.process import data_tranform2d
from skhyper.utils._data_checks import _data_checks
from skhyper.utils._plot import _plot_cluster


class AgglomerativeClustering:
    def __init__(self, n_clusters, affinity='euclidean', memory=None, connectivity=None,
                 compute_full_tree='auto', linkage='ward', pooling_func=np.mean):
        self.data = None
        self._shape = None
        self._dimensions = None

        # clustering outputs
        self.labels = None
        self.data_clusters = None

        # sklearn optional AgglomerativeClustering arguments
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.pooling_func = pooling_func

    def fit(self, data):
        self.data = data
        self._shape, self._dimensions = _data_checks(self.data)

        data2d = data_tranform2d(self.data)
        agglomerative_model = _sklearn_agglomerative_clustering(n_clusters=self.n_clusters, affinity=self.affinity,
                                                                memory=self.memory, connectivity=self.connectivity,
                                                                compute_full_tree=self.compute_full_tree,
                                                                linkage=self.linkage,
                                                                pooling_func=self.pooling_func).fit(data2d)

        if self._dimensions == 3:
            labels = np.reshape(agglomerative_model.labels_, (self._shape[0], self._shape[1]))
            labels += 1
            data_clusters = [0]*self.n_clusters
            for cluster in range(self.n_clusters):
                data_clusters[cluster] = np.zeros((self._shape[0], self._shape[1], self._shape[2]))
                for spectral_pixel in range(self._shape[2]):
                    data_clusters[cluster][:, :, spectral_pixel] = np.multiply(
                        self.data[:, :, spectral_pixel], np.where(labels == cluster + 1, labels, 0) / (cluster + 1))

        elif self._dimensions == 4:
            labels = np.reshape(agglomerative_model.labels_, (self._shape[0], self._shape[1], self._shape[2]))
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
