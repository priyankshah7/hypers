import numpy as np
from scipy.signal import savgol_filter as _savgol_filter

from skhyper.process import data_shape


def savgol_filter(data, window_length=5, poly_order=3):
    """
    :param data:
    :return smoothedData:

    Smooths data using the Savitzky-Golay filter
    """
    shape, dimensions = data_shape(data)

    if dimensions == 1:
        smoothed_data = np.zeros((shape[0]))
        smoothed_data = _savgol_filter(data, window_length, poly_order)

        return smoothed_data

    elif dimensions == 2:
        smoothed_data = np.zeros((shape[0], shape[1]))
        for xrow in range(shape[0]):
            smoothed_data[xrow, :] = _savgol_filter(data[xrow, :], window_length, poly_order)

        return smoothed_data

    elif dimensions == 3:
        smoothed_data = np.zeros((shape[0], shape[1], shape[2]))
        for xrow in range(shape[0]):
            for yrow in range(shape[1]):
                smoothed_data[xrow, yrow, :] = _savgol_filter(data[xrow, yrow, :], window_length, poly_order)

        return smoothed_data

    elif dimensions == 4:
        smoothed_data = np.zeros((shape[0], shape[1], shape[2], shape[3]))
        for xrow in range(shape[0]):
            for yrow in range(shape[1]):
                for zrow in range(shape[2]):
                    smoothed_data[xrow, yrow, zrow, :] = _savgol_filter(data[xrow, yrow, zrow, :], window_length, poly_order)

        return smoothed_data
