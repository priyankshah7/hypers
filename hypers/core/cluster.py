import numpy as np
from typing import List
from collections import namedtuple
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering

import hypers as hp
from hypers.exceptions import warning_insufficient_samples
from hypers.core import spatial_weights_matrix

__all__ = ['asc_hierarchical']

ASCHierarchical = namedtuple(
    'ASCHierarchical',
    ['clusters', 'labels', 'region_labels', 'region_spectra']
)


def asc_hierarchical(X: 'hp.hparray', k: int = 5, n_regions: int = 20,
                     n_clusters: int = None) -> ASCHierarchical:
    data = X.collapse()
    data_mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
    data_std = np.std(data, axis=1).reshape(data.shape[0], 1)

    # standardise prior to cluster
    data -= data_mean
    data /= data_std

    # cluster
    if X.nsamples < X.nspectral:
        warning_insufficient_samples(X.nsamples, X.nspectral)

    w = spatial_weights_matrix(image_shape=X.nspatial, k=k)
    agg = AgglomerativeClustering(n_clusters=n_regions, connectivity=w.sparse, linkage='ward')
    agg.fit(data)
    region_labels = agg.labels_.reshape(X.nspatial)
    region_spectra = _get_spectra_from_cluster_labels(X, region_labels)
    clusters = _cluster_spectra_cosine_distances(region_spectra, n_clusters=n_clusters)

    asc = ASCHierarchical(
        clusters=clusters,
        labels=None,
        region_labels=region_labels,
        region_spectra=region_spectra
    )

    return asc


def _get_spectra_from_cluster_labels(data: hp.hparray, labels: np.ndarray) -> np.ndarray:
    labels += 1
    unique_vals = np.unique(labels.reshape(labels.size, 1))
    spectra = np.zeros((data.shape[-1], unique_vals.size))
    for index, val in enumerate(unique_vals):
        mask = np.where(labels == val, labels, 0) / val
        spectra[:, index] = np.squeeze(np.mean(np.mean(data * mask.reshape(mask.shape + (1,)), 1), 0))

    return spectra


def _cluster_spectra_cosine_distances(spectra: np.ndarray, n_clusters: int = None,
                                      max_distance: float = 0.2) -> List[np.ndarray]:
    n_spectra = spectra.shape[-1]
    clusters = []
    for index in range(n_spectra):
        spectrum = np.squeeze(spectra[:, index])
        spectrum -= np.mean(spectrum)
        spectrum /= np.max(np.abs(spectrum))

        if index == 0:
            clusters.append(spectrum)

        else:
            distances = []
            for cluster in clusters:
                dist = cosine(spectrum, cluster)
                distances.append(dist)

            dist_index_match = []
            for dist_index, dist in enumerate(distances):
                if dist < max_distance:
                    dist_index_match.append((dist_index, dist))

            if len(dist_index_match) == 0:
                clusters.append(spectrum)
            elif len(dist_index_match) == 1:
                clusters[dist_index_match[0][0]] = np.mean(
                    [clusters[dist_index_match[0][0]], spectrum],
                    axis=0
                )
            else:
                pass

    if n_clusters is not None:
        pass

    return clusters


def _label_data_from_clusters(X: 'hp.hparray', clusters: List[np.ndarray]) -> np.ndarray:
    pass
