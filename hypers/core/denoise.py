import numpy as np
from collections import namedtuple
from typing import Tuple, List
from scipy.stats import normaltest
from pysal.lib.weights import W
from pysal.explore.esda import Moran

import hypers as hp
from hypers.exceptions import warning_insufficient_samples
from hypers.core import spatial_weights_matrix

__all__ = ['low_rank_approximation']

LRA = namedtuple(
    'LRA',
    ['data', 'morans_i', 'accept_index', 'pc_ims', 'pc_specs']
)


def low_rank_approximation(X: 'hp.hparray', k: int = 5, p_sig: float = 0.05,
                           return_pc: bool = False) -> LRA:
    data = X.collapse()
    nspatial = X.nspatial
    data_mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
    data_std = np.std(data, axis=1).reshape(data.shape[0], 1)

    # standardise prior to denoise
    data -= data_mean
    data /= data_std

    # denoise
    w = spatial_weights_matrix(image_shape=nspatial, k=k)
    pc_ims, pc_specs = _calculate_principal_components(data)
    if X.nsamples >= X.nspectral:
        assert pc_ims.shape == (X.nsamples, X.nspectral)
        assert pc_specs.shape == (X.nspectral, X.nspectral)
    elif X.nsamples < X.nspectral:
        warning_insufficient_samples(X.nsamples, X.nspectral)

    moran_objs = _pc_spatial_moran_objs(pc_ims, w)
    accept_index, reject_index = _pc_spatial_acceptance_check(moran_objs, p_sig=p_sig)

    spectral_distributions = _pc_spectral_signal_distributions(pc_specs, reject_index, nruns=500)
    spectral_accept_index = _pc_spectral_acceptance_check(spectral_distributions, p_sig=p_sig)

    new_accept_index = [reject_index[index] for index in spectral_accept_index]
    for val in new_accept_index:
        accept_index.append(val)
    accept_index.sort()

    data = pc_ims[:, accept_index] @ pc_specs[:, accept_index].T

    # "de-standardise"
    data /= data_std
    data += data_mean

    data = hp.hparray(data.reshape(X.shape))
    moran_vals = _pc_spatial_moran_vals(moran_objs)
    lra = LRA(
        data=data,
        morans_i=moran_vals,
        accept_index=accept_index,
        pc_ims=pc_ims if return_pc else None,
        pc_specs=pc_specs if return_pc else None
    )

    return lra


def _calculate_principal_components(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    ims = u * s
    specs = vh.T
    assert ims.shape[-1] == specs.shape[-1]
    return ims, specs


def _pc_spatial_moran_objs(pc_ims: np.ndarray, w: W) -> List[Moran]:
    assert len(pc_ims.shape) == 2
    n_components = pc_ims.shape[-1]
    moran_objs = []
    for index in range(n_components):
        im = pc_ims[:, index]
        val = Moran(im.reshape(im.size, 1), w)
        moran_objs.append(val)

    return moran_objs


def _pc_spatial_moran_vals(moran_objs: List[Moran]) -> List[float]:
    n_components = len(moran_objs)
    moran_vals = [moran_objs[index].I for index in range(n_components)]
    return moran_vals


def _pc_spatial_acceptance_check(moran_objs: List[Moran], p_sig: float = 0.05) -> Tuple[List, List]:
    n_components = len(moran_objs)
    accept_index = []
    reject_index = []
    for index in range(n_components):
        p_rand = moran_objs[index].p_rand
        if p_rand < p_sig:
            accept_index.append(index)
        else:
            reject_index.append(index)

    return accept_index, reject_index


def _pc_spectral_signal_distributions(pc_specs: np.ndarray, reject_index: List[int],
                                      nruns: int = 500) -> List[np.ndarray]:
    distributions = []
    for index in range(len(reject_index)):
        signal = np.squeeze(pc_specs[:, index])
        dist = _single_signal_distribution(signal, nruns)
        distributions.append(dist)

    return distributions


def _single_signal_distribution(signal: np.ndarray, nruns: int = 500) -> np.ndarray:
    bootstrap_means = []
    for _ in range(nruns):
        mean = np.mean(np.random.choice(signal, size=int(signal.size / 2)))
        bootstrap_means.append(mean)

    return np.array(bootstrap_means)


def _pc_spectral_acceptance_check(distributions: List[np.ndarray], p_sig: float = 0.05) -> List[int]:
    n_dists = len(distributions)
    accept_index = []
    for index in range(n_dists):
        _, pval = normaltest(np.squeeze(distributions[index]))
        if pval < p_sig:
            accept_index.append(index)

    return accept_index
