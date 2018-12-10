import numpy as np 


def _data_normalization(X):
    """Normalize the hyperspectral data

    Normalizes the hyperspectral data by subtracting the mean spectrum of the
    data from each pixel.
        """
    if X.ndim == 3:
        mean_spectrum = np.squeeze(np.mean(np.mean(X.data, 1), 0))

        for xpix in range(X.shape[0]):
            for ypix in range(X.shape[1]):
                X.data[xpix, ypix, :] -= mean_spectrum

    elif X.ndim == 4:
        mean_spectrum = np.squeeze(np.mean(np.mean(np.mean(X.data, 2), 1), 0))

        for xpix in range(X.shape[0]):
            for ypix in range(X.shape[1]):
                for zpix in range(X.shape[2]):
                    X.data[xpix, ypix, zpix, :] -= mean_spectrum
