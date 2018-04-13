import numpy as np

from skhyper.process import data_shape


def spectral_normalisation(data, spectrum):
    shape, dimensions = data_shape(data)
    spectrum_shape, spectrum_dimensions = data_shape(spectrum)

    if spectrum_dimensions != 1:
        raise TypeError('Spectrum to normalize by must be 1-dimensional')

    if dimensions == 3:
        if shape[2] != spectrum_shape[0]:
            raise ValueError('The number of spectral points must be the same for both the data and spectrum')

        normalized_data = np.zeros((shape[0], shape[1], shape[2]))

        for x in range(shape[0]):
            for y in range(shape[1]):
                normalized_data[x, y, :] = data[x, y, :] / spectrum

    elif dimensions == 4:
        if shape[3] != spectrum_shape[0]:
            raise ValueError('The number of spectral points must be the same for both the data and spectrum')

        normalized_data = np.zeros((shape[0], shape[1], shape[2], shape[3]))

        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    normalized_data[x, y, z, :] = data[x, y, z, :] / spectrum

    else:
        raise TypeError('Data must be 3- or 4-dimensional.')

    return normalized_data
