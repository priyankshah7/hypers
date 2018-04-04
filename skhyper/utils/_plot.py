import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from skhyper.utils import HyperanalysisError


# TODO change to plotly to allow interactivity in Jupyter notebooks
def _plot_cluster(cluster_data, dim=3):
    if dim == 3:
        plt.figure(facecolor='w')
        grid = GridSpec(len(cluster_data), 2, width_ratios=[1, 2])
        for cluster in range(len(cluster_data)):
            plt.subplot(grid[(2 * cluster)])
            plt.imshow(np.squeeze(np.mean(cluster_data[cluster], 2)))
            plt.axis('off')
            plt.subplot(grid[(2 * cluster) + 1])
            plt.plot(np.squeeze(np.mean(np.mean(cluster_data[cluster], 1), 0)))
        plt.tight_layout()
        plt.show()

    # NOTE at the moment, this will only display the first z layer
    elif dim == 4:
        plt.figure(facecolor='w')
        grid = GridSpec(len(cluster_data), 2, width_ratios=[1, 2])
        for cluster in range(len(cluster_data)):
            plt.subplot(grid[(2 * cluster)])
            plt.imshow(np.squeeze(np.mean(cluster_data[cluster][:, :, 0, :], 2)))
            plt.axis('off')
            plt.subplot(grid[(2 * cluster) + 1])
            plt.plot(np.squeeze(np.mean(np.mean(cluster_data[cluster][:, :, 0, :], 1), 0)))
        plt.tight_layout()
        plt.show()


def _plot_decomposition(plot_range, images, spectra, dim=3):
    if not isinstance(plot_range, tuple):
        raise HyperanalysisError('plot_range must be a tuple containing two elements e.g. plot_range=(0, 5)')

    if not len(plot_range) == 2:
        raise HyperanalysisError('plot_range must be a tuple containing two elements e.g. plot_range=(0, 5)')

    if dim == 3:
        comps_required = plot_range[1] - plot_range[0]
        plt.figure(facecolor='w')
        grid = GridSpec(comps_required, 2, width_ratios=[1, 2])
        for component in range(plot_range[0], plot_range[1]):
            plt.subplot(grid[(2 * (component - plot_range[0]))])
            plt.imshow(images[:, :, component])
            plt.colorbar()
            plt.axis('off')
            plt.subplot(grid[(2 * (component - plot_range[0])) + 1])
            plt.plot(spectra[:, component])
        plt.show()

    if dim == 4:
        comps_required = plot_range[1] - plot_range[0]
        plt.figure(facecolor='w')
        grid = GridSpec(comps_required, 2, width_ratios=[1, 2])
        for component in range(plot_range[0], plot_range[1]):
            plt.subplot(grid[(2 * (component - plot_range[0]))])
            plt.imshow(images[:, :, 0, component])
            plt.colorbar()
            plt.axis('off')
            plt.subplot(grid[(2 * (component - plot_range[0])) + 1])
            plt.plot(spectra[:, component])
        plt.show()
