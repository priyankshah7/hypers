import numpy as np
import hypers as hp
import matplotlib.pyplot as plt


def _ucls(X: 'hp.Dataset',
          spectra: np.ndarray,
          plot: bool = False,
          return_arrs: bool = True) -> np.ndarray:

    x_inverse = np.linalg.pinv(spectra)
    im = np.dot(x_inverse, X.flatten().T).T.reshape(X.shape[:-1] + (spectra.shape[-1],))

    if plot:
        n_rows = np.ceil(spectra.shape[-1] / 3)
        if X.ndim == 3:
            for endm in range(spectra.shape[-1]):
                plt.subplot(n_rows, 3, endm+1)
                plt.imshow(np.squeeze(im[..., endm]))
                plt.title('Map ' + str(endm+1))
                plt.axis('off')
        elif X.ndim == 4:
            for endm in range(spectra.shape[-1]):
                plt.subplot(n_rows, 3, endm+1)
                plt.imshow(np.mean(np.squeeze(im[..., endm]), -1))
                plt.title('Map' + str(endm + 1))
                plt.axis('off')
        plt.show()

    if return_arrs:
        return im
