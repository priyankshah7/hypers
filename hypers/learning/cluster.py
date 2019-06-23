import numpy as np
import hypers as hp
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# TODO Add hierarchical (agglomerative) clustering

class cluster:
    def __init__(self, X: 'hp.core.hparray'):
        self.kmeans = kmeans(X)
        self.hierarchical = None


class kmeans:
    def __init__(self, X: 'hp.hparray'):
        self.X = X
        self.labels = None
        self.spcs = None

    def calculate(self, n_clusters: int = 4,
                  decomposed: bool = False,
                  pca_comps: int = 10, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

        mdl_cluster = KMeans(n_clusters=n_clusters, **kwargs)
        if decomposed:
            print('Clustering with the first', str(pca_comps), ' PCA components')
            mdl_pca = PCA(n_components=pca_comps)
            comps = mdl_pca.fit_transform(self.X.flatten())
            mdl_cluster.fit(comps)
            spcs = mdl_pca.inverse_transform(mdl_cluster.cluster_centers_)

        else:
            mdl_cluster.fit(self.X.collapse())
            spcs = mdl_cluster.cluster_centers_

        self.labels = mdl_cluster.labels_.reshape(self.X.data.shape[:-1])
        self.spcs = np.transpose(spcs)

        return self.labels, self.spcs

    # TODO Coordinate colors and cluster numbers of retrieved clusters and plots

    def plot_image(self, z: int = None):
        if self.X.ndim == 3:
            plt.imshow(self.labels)

        elif self.X.ndim == 3:
            plt.imshow(np.squeeze(self.labels[..., z]))

        plt.colorbar()
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    def plot_spectrum(self):
        for cluster in range(self.spcs.shape[-1]):
            plt.plot(self.spcs[..., cluster], label='Clust. '+str(cluster+1))
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_grid(self, z: int = None):
        plt.subplot(121)
        if self.X.ndim == 3:
            plt.imshow(self.labels)
        elif self.X.ndim == 4:
            plt.imshow(np.squeeze(self.labels[..., z]))
        plt.colorbar()
        plt.grid(False)
        plt.subplot(122)
        for cluster in range(self.spcs.shape[-1]):
            plt.plot(self.spcs[..., cluster], label='Clust. ' + str(cluster + 1))
        plt.legend()
        plt.tight_layout()
        plt.show()
