import hypers as hp
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter


def _data_smoothen(X: 'hp.Dataset', **kwargs):

    if X.smoothing == 'savitzky_golay':
        if X.ndim == 3:
            for xpix in range(X.shape[0]):
                for ypix in range(X.shape[1]):
                    X.data[xpix, ypix, :] = savgol_filter(X.data[xpix, ypix, :], **kwargs)

        elif X.ndim == 4:
            for xpix in range(X.shape[0]):
                for ypix in range(X.shape[1]):
                    for zpix in range(X.shape[2]):
                        X.data[xpix, ypix, zpix, :] = savgol_filter(X.data[xpix, ypix, zpix, :], **kwargs)

    elif X.smoothing == 'gaussian_filter':
        if X.ndim == 3:
            for xpix in range(X.shape[0]):
                for ypix in range(X.shape[1]):
                    X.data[xpix, ypix, :] = gaussian_filter(X.data[xpix, ypix, :], **kwargs)

        elif X.ndim == 4:
            for xpix in range(X.shape[0]):
                for ypix in range(X.shape[1]):
                    for zpix in range(X.shape[2]):
                        X.data[xpix, ypix, zpix, :] = gaussian_filter(X.data[xpix, ypix, zpix, :], **kwargs)
