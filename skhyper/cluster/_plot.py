import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns

sns.set()


def _plot_components(images, specs, title):
    if len(images) != len(specs):
        raise IndexError('The number of clusters for the images and specs must be the same')

    num_components = len(images)

    sns.set_style('dark')
    fig = plt.figure()
    ax = [0] * (num_components*2)
    gs = GridSpec(num_components, 2, width_ratios=[1, 3])
    for comp in range(num_components):
        # Plotting images
        ax[2*comp] = plt.subplot(gs[2*comp])
        axim = ax[2 * comp].imshow(images[comp])
        cb = plt.colorbar(axim, ax=ax[2 * comp])
        tick_locator = ticker.MaxNLocator(nbins=3)
        cb.locator = tick_locator
        cb.update_ticks()
        plt.xticks([])
        plt.yticks([])

        # Plotting spectra
        ax[2*comp + 1] = plt.subplot(gs[2*comp + 1])
        ax[2*comp + 1].plot(specs[comp])
    plt.suptitle(title)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    plt.show()
