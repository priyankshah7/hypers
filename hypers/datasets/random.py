import numpy as np
from collections import namedtuple
from typing import Tuple, List
from sklearn.datasets import make_blobs

import hypers as hp

__all__ = ['make_hsi', 'make_noise', 'make_random']

HSIRandom = namedtuple(
    'HSIRandom',
    ['data', 'unique_spectra', 'labels']
)


def make_noise(nspatial: Tuple[int], nspectral: int, noise_level: float = 0.1) -> HSIRandom:
    """
    Generate a simulated HSI array with no spatial/spectral correlation.

    Parameters
    ----------
    nspatial: Tuple[int]
        A tuple of 2 values denoting the spatial size.
    nspectral: int
        The number of spectral bands.
    noise_level: float
        The level of noise for spectra. A value of 0 will include
        no noise. The added noise is additive, white Gaussian noise.

    Returns
    -------
    HSIRandom
        An object containing the following attributes:
            - data: A hp.hparray of the generated HSI
            - unique_spectra: set to None.
            - labels: set to None.
    """
    assert len(nspatial) == 2
    hsi_shape = nspatial + (nspectral,)
    data = np.random.normal(size=hsi_shape)
    data *= noise_level
    data = hp.array(data)

    hsi = HSIRandom(
        data=data,
        unique_spectra=None,
        labels=None
    )

    return hsi


def make_random(nspatial: Tuple[int], nspectral: int, noise_level: float = 0.0,
                n_clusters: int = 5) -> HSIRandom:
    """
    Generate a simulated HSI array with no spatial correlation.

    Parameters
    ----------
    nspatial: Tuple[int]
        A tuple of 2 values denoting the spatial size.
    nspectral: int
        The number of spectral bands.
    noise_level: float
        The level of noise for spectra. A value of 0 will include
        no noise. The added noise is additive, white Gaussian noise.
    n_clusters: int
        The number of unique clusters.

    Returns
    -------
    HSIRandom
        An object containing the following attributes:
            - data: A hp.hparray of the generated HSI
            - unique_spectra: A list of each unique spectra corresponding the n_clusters.
            - labels: An np.ndarray image containing the labels for each spatial pixel.
    """
    assert len(nspatial) == 2
    n_samples = np.prod(nspatial)
    hsi_shape = nspatial + (nspectral,)

    data, labels, unique_spectra = make_blobs(
        n_samples=n_samples, n_features=nspectral, centers=n_clusters, return_centers=True
    )
    data += np.random.normal(size=(nspectral,)) * noise_level
    data = data.reshape(hsi_shape)
    data = hp.array(data)
    labels = labels.reshape(nspatial)

    hsi = HSIRandom(
        data=data,
        unique_spectra=unique_spectra,
        labels=labels
    )

    return hsi


def make_hsi(nspatial: Tuple[int], nspectral: int, n_clusters: int, npeaks_min: int,
             npeaks_max: int, peak_amplitudes_min: int, peak_amplitudes_max: int, peak_std_min: int,
             peak_std_max: int, noise_level: float, n_circles: int, circle_radius_min: int,
             circle_radius_max: int, circle_allow_overlap: bool = False,
             background_noise_level: float = None) -> HSIRandom:
    """
    Generate a simulated HSI array.

    Parameters
    ----------
    nspatial: Tuple[int]
        A tuple of 2 values denoting the spatial size.
    nspectral: int
        The number of spectral bands.
    n_clusters: int
        The number of unique clusters.
    npeaks_min: int
        The minimum number of peaks in any unique spectrum.
    npeaks_max: int
        The maximum number of peaks in any unique spectrum.
    peak_amplitudes_min: int
        The minimum amplitude of any peaks in any spectrum.
    peak_amplitudes_max: int
        The maximum amplitude of any peaks in any spectrum.
    peak_std_min: int
        The minimum standard deviation of any peak in any spectrun.
    peak_std_max: int
        The maximum standard deviation of any peak in any spectrum.
    noise_level: float
        The level of noise for spectra with peaks. A value of 0 will include
        no noise. The added noise is additive, white Gaussian noise.
    n_circles: int
        The number of circles to include in the spatial component of the HSI.
    circle_radius_min: int
        The minimum radius of any circle in the HSI.
    circle_radius_max: int
        The maximum radius of any circle in the HSI.
    circle_allow_overlap: bool
        Whether to allow circles to overlap or not.
    background_noise_level: float
        The level of noise for background spectra. A value of 0 will include
        no noise. The added noise is additive, white Gaussian noise.

    Returns
    -------
    HSIRandom
        An object containing the following attributes:
            - data: A hp.hparray of the generated HSI
            - unique_spectra: A list of each unique spectra corresponding the n_clusters.
            - labels: An np.ndarray image containing the labels for each spatial pixel.
    """
    assert len(nspatial) == 2
    if background_noise_level is None:
        background_noise_level = noise_level

    xvals = np.arange(nspectral)
    spectra = []
    for index in range(n_clusters):
        npeaks = np.random.randint(npeaks_min, npeaks_max)
        amplitudes = [np.random.randint(peak_amplitudes_min, peak_amplitudes_max) for _ in range(npeaks)]
        stds = [np.random.randint(peak_std_min, peak_std_max) for _ in range(npeaks)]
        positions = np.random.choice(xvals[3:-3], size=npeaks, replace=False)
        spectrum = generate_signal(
            xvals=xvals, peak_amplitudes=amplitudes, peak_means=positions, peak_std=stds,
            noise_level=noise_level
        )
        spectra.append(spectrum)

    circle_radiuses = [np.random.randint(circle_radius_min, circle_radius_max) for _ in range(n_circles)]
    circle_positions = [
        (np.random.randint(1, nspatial[0] - 1), np.random.randint(1, nspatial[1] - 1)) for _ in range(n_circles)
    ]
    labels = generate_circles(
        positions=circle_positions, radiuses=circle_radiuses, shape=nspatial, n_clusters=n_clusters,
        allow_overlap=circle_allow_overlap
    )

    hsi_shape = nspatial + (nspectral,)
    data = np.zeros(shape=hsi_shape)
    for x in range(nspatial[0]):
        for y in range(nspatial[1]):
            cluster = int(labels[x, y])
            if not cluster == 0:
                data[x, y] = spectra[cluster - 1]
            else:
                noise = np.random.normal(size=xvals.size)
                data[x, y] = noise * background_noise_level

    data = hp.array(data)

    hsi = HSIRandom(
        data=data,
        unique_spectra=spectra,
        labels=labels
    )

    return hsi


def generate_signal(xvals: np.ndarray, peak_amplitudes: List[int], peak_means: List[int],
                    peak_std: List[float], noise_level: float = 0.0) -> np.ndarray:
    """
    Generate a signal with Gaussian peaks.

    Parameters
    ----------
    xvals: np.ndarray
        An array of the spectral bands.
    peak_amplitudes: List[int]
        A list of amplitudes for the peaks in the spectrum.
    peak_means: List[int]
        A list of positions for the peaks in the spectrum.
    peak_std: List[float]
        A list of the standard deviation for the peaks in the spectrum.
    noise_level: float
        The level of noise. A value of 0 will include no noise.
        The added noise is additive, white Gaussian noise.

    Returns
    -------
    np.ndarray
        The generated signal.
    """
    assert len(peak_amplitudes) == len(peak_means)
    assert len(peak_amplitudes) == len(peak_std)

    n_peaks = len(peak_amplitudes)
    spectrum = np.zeros(shape=(xvals.size,))
    for peak in range(n_peaks):
        A = peak_amplitudes[peak]
        mu = peak_means[peak]
        sigma = peak_std[peak]
        spectrum += A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(xvals - mu) ** 2 / (2 * sigma ** 2))

    noise = np.random.normal(size=xvals.size)
    spectrum += noise * noise_level

    return spectrum


def generate_circles(positions: List[tuple], radiuses: List[int], shape: tuple = (100, 100),
                     n_clusters: int = 2, allow_overlap: bool = True) -> np.ndarray:
    """
    Generate image with random circles.

    Parameters
    ----------
    positions: List[tuple]
        A list of tuples (of len 2) denoting the coordinates of the centers for
        the circles in the image.
    radiuses: List[int]
        A list of radiuses for the circles in the image.
    shape: tuple
        The shape of the image given as a tuple (of len 2)
    n_clusters: int
        The number of clusters to assign.
    allow_overlap: bool
        Whether or not to allow the overlap of circles when assigning positions.

    Returns
    -------
    np.ndarray
        An array of the generated image with the values of each pixel
        reflecting the assigned cluster.
    """
    # TODO At the moment, if n circles overlap then only (n_clusters - n) circles are printed
    assert len(positions) == len(radiuses)
    assert len(shape) == 2

    image = np.zeros(shape=shape)
    chosen_coords = []
    for index, position in enumerate(positions):
        use = True
        chosen_cluster = np.random.randint(1, n_clusters + 1)
        xs = []
        ys = []
        for x, y in np.ndindex(shape):
            if np.sqrt((x - position[0]) ** 2 + (y - position[1]) ** 2) <= radiuses[index]:
                xs.append(x)
                ys.append(y)

        if not allow_overlap:
            zipped_coords = list(zip(xs, ys))
            if any(coord in zipped_coords for coord in chosen_coords):
                use = False

            if use:
                image[xs, ys] = chosen_cluster
                for coord in zipped_coords:
                    chosen_coords.append(coord)
        else:
            image[xs, ys] = int(chosen_cluster)

    return image
