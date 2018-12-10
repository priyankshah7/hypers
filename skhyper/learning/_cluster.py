import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering,
    DBSCAN, Birch
)

CLUSTER_TYPES = (
    KMeans,
    # AffinityPropagation,
    # MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    # DBSCAN,
    # Birch
)


def _data_cluster(X, mdl, decomposed=False, pca_comps=4):
    if type(mdl) not in CLUSTER_TYPES:
        raise TypeError('Must pass a sklearn cluster class. Refer to documentation.')

    X.mdl_cluster = mdl
    n_clusters = X.mdl_cluster.get_params()['n_clusters']
    if decomposed:
        print('Clustering with the first ' + str(pca_comps) + ' PCA components.')
        mdl_pca = PCA(n_components=pca_comps)
        comps = mdl_pca.fit_transform(X.flatten())
        X.mdl_cluster.fit(comps)
        labels = X.mdl_cluster.labels_.reshape(X.data.shape[:-1])
            
        try:
            specs = mdl_pca.inverse_transform(X.mdl_cluster.cluster_centers_)
        except AttributeError:
            specs = np.zeros((n_clusters, X.data.shape[-1]))
            lbls = labels + 1
            for cluster_number in range(n_clusters):
                msk = np.zeros(X.data.shape)
                for spectral_point in range(X.data.shape[-1]):
                    msk[..., spectral_point] = np.multiply(
                        X.data[..., spectral_point], 
                        np.where(lbls==cluster_number+1, lbls, 0)/(cluster_number+1)
                    )
                    
                if X.ndim == 3:
                    specs[cluster_number, :] = np.squeeze(np.mean(np.mean(msk, 1), 0))
                elif X.ndim == 4:
                    specs[cluster_number, :] = np.squeeze(np.mean(np.mean(np.mean(msk, 2), 1), 0))
            # specs = mdl_pca.inverse_transform(specs.transpose())
            # TODO Check whether you are getting the correct PCA reduced clustering here

    else:
        X.mdl_cluster.fit(X.flatten())
        labels = X.mdl_cluster.labels_.reshape(X.data.shape[:-1])

        try:
            specs = X.mdl_cluster.cluster_centers_
        except AttributeError:
            specs = np.zeros((n_clusters, X.data.shape[-1]))
            lbls = labels + 1
            for cluster_number in range(n_clusters):
                msk = np.zeros(X.data.shape)
                for spectral_point in range(X.data.shape[-1]):
                    msk[..., spectral_point] = np.multiply(
                        X.data[..., spectral_point], 
                        np.where(lbls==cluster_number+1, lbls, 0)/(cluster_number+1)
                    )
                    
                if X.ndim == 3:
                    specs[cluster_number, :] = np.squeeze(np.mean(np.mean(msk, 1), 0))
                elif X.ndim == 4:
                    specs[cluster_number, :] = np.squeeze(np.mean(np.mean(np.mean(msk, 2), 1), 0))

    return labels, specs
