import numpy as np
import matplotlib.pyplot as plt


# TODO change to plotly to allow interactivity in Jupyter notebooks
def _plot_cluster(cluster_data, dim=3):
    if dim == 3:
        plt.figure(facecolor='w')
        for cluster in range(len(cluster_data)):
            plt.subplot(len(cluster_data), 2, (2 * cluster) + 1)
            plt.imshow(np.squeeze(np.mean(cluster_data[cluster], 2)))
            plt.axis('off')
            plt.subplot(len(cluster_data), 2, (2 * cluster) + 2)
            plt.plot(np.squeeze(np.mean(np.mean(cluster_data[cluster], 1), 0)))
        plt.tight_layout()
        plt.show()

    # NOTE at the moment, this will only display the first z layer
    elif dim == 4:
        plt.figure(facecolor='w')
        for cluster in range(len(cluster_data)):
            plt.subplot(len(cluster_data), 2, (2 * cluster) + 1)
            plt.imshow(np.squeeze(np.mean(cluster_data[cluster][:, :, 0, :], 2)))
            plt.axis('off')
            plt.subplot(len(cluster_data), 2, (2 * cluster) + 2)
            plt.plot(np.squeeze(np.mean(np.mean(cluster_data[cluster][:, :, 0, :], 1), 0)))
        plt.tight_layout()
        plt.show()
