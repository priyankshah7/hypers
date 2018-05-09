"""
Spectral normalization
"""
import numpy as np

from skhyper.process import Process


def normalize(X, spectrum):
    """ Normalize spectra in the hyperspectral data

    Normalizes the spectra in the hyperspectral data. The data is modified directly in the object,
    i.e. a new object is not created (this is to limit memory usage).

    Parameters
    ----------
    X : object, Process instance
        The object X containing the hyperspectral array.

    spectrum : array
        Must be 1-dimensional and the same size as the spectra in X.

    """
    if not isinstance(X, Process):
        raise TypeError('Data needs to be passed to skhyper.process.Process first')

    elif type(spectrum) != np.ndarray:
        raise TypeError('Spectrum to normalize by must be a 1-d numpy array')

    elif len(spectrum.shape) != 1:
        raise TypeError('Spectrum to normalize by must be a 1-d numpy array')

    elif X.shape[-1] != spectrum.shape[0]:
        raise ValueError('The spectral range of the spectrum and hyperspectral data need to be the same')

    else:
        if X.n_dimension == 3:
            for _x in range(X.shape[0]):
                for _y in range(X.shape[1]):
                    X.data[_x, _y, :] /= spectrum

        elif X.n_dimension == 4:
            for _x in range(X.shape[0]):
                for _y in range(X.shape[1]):
                    for _z in range (X.shape[2]):
                        X.data[_x, _y, _z, :] /= spectrum

        X._initialize()
        print('Hyperspectral data normalized.')
