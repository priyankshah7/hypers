import numpy as np
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def plot(X, **kwargs):
    if X.ndim == 3:
        plt.plot(np.squeeze(np.mean(np.mean(X[...], 1), 0)), **kwargs)

    elif X.ndim == 4:
        plt.plot(np.squeeze(np.mean(np.mean(np.mean(X[...], 2), 1), 0)), **kwargs)


def imshow(X, **kwargs):
    if X.ndim == 3:
        plt.imshow(np.squeeze(np.mean(X[...], 2)), **kwargs)

    elif X.ndim == 4:
        plt.imshow(np.squeeze(np.mean(X[...], 3)), **kwargs)


def components(images=None, spectra=None):
    if images is not None and spectra is not None:
        if len(images) != len(spectra):
            raise IndexError('The number of components for the images and specs must be the same')

        num_components = len(images)

        ax = [0] * (num_components*2)
        gs = GridSpec(num_components, 2, width_ratios=[1, 3])
        for comp in range(num_components):
            # Plotting images
            ax[2*comp] = plt.subplot(gs[2*comp])
            axim = ax[2*comp].imshow(images[comp])
            cb = plt.colorbar(axim, ax=ax[2 * comp])
            tick_locator = ticker.MaxNLocator(nbins=3)
            cb.locator = tick_locator
            cb.update_ticks()
            plt.xticks([])
            plt.yticks([])

            # Plotting spectra
            ax[2*comp + 1] = plt.subplot(gs[2*comp + 1])
            ax[2*comp + 1].plot(spectra[comp])

    elif images is not None and spectra is None:
        num_components = len(images)

        ax = [0] * num_components
        gs = GridSpec(num_components//4 + 1, 4)
        for comp in range(num_components):
            ax[comp] = plt.subplot(gs[comp])
            axim = ax[comp].imshow(images[comp])
            cb = plt.colorbar(axim, ax=ax[comp])
            tick_locator = ticker.MaxNLocator(nbins=3)
            cb.locator = tick_locator
            cb.update_ticks()
            plt.xticks([])
            plt.yticks([])

    elif images is None and spectra is not None:
        num_components = len(spectra)

        ax = [0] * num_components
        gs = GridSpec(num_components//2 + 1, 2)
        for comp in range(num_components):
            ax[comp] = plt.subplot(gs[comp])
            ax[comp].plot(spectra[comp])

    else:
        raise TypeError('You must pass either the image or spectral components or both.')

