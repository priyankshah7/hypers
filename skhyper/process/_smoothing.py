"""
Spectral smoothing
"""
from scipy.signal import savgol_filter as _savgol_filter

from skhyper.process import Process


def savgol_filter(X, window_length=5, polyorder=3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0):
    """ Savitzky-Golay filter
    
    Applies a Savitzky-Golay filter to all the spectral components in the hyperspectral array.
    The data is modified directly in the object, i.e. a new object is not created
    (this is to limit memory usage).

    Parameters
    ----------
    X : object, Process instance
        The object X containing the hyperspectral array.

    window_length : int, optional (default: 5)
        The length of the filter window (i.e. the number of coefficients). `window_length`
        must be a positive odd integer.

    polyorder : int, optional (default: 3)
        The order of the polynomial used to fit the samples. `polyorder` must be less than
        `window_length`.

    deriv : int, optional (default: 0)
        The order of the derivative to compute. This must be a nonnegative intefer. The
        default it 0, which means to filter the data without differentiating.

    delta : float, optional (default: 1.0)
        The spacing of the samples to which the data will be applied. This is only used
        if `deriv`>0.

    axis : int, optional (default: -1)
        The axis of the array along which the filter is to be applied

    mode : str, optional (default: 'interp')
        Must be ‘mirror’, ‘constant’, ‘nearest’, ‘wrap’ or ‘interp’. This determines
        the type of extension to use for the padded signal to which the filter is
        applied. When mode is ‘constant’, the padding value is given by `cval`.
        See the Notes for more details on ‘mirror’, ‘constant’, ‘wrap’, and
        ‘nearest’. When the ‘interp’ mode is selected (the default), no
        extension is used. Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is used to
        evaluate the last `window_length` // 2 output values.

    cval : scalar, optional (default: 0.0)
        Value to fill past the edges of the input if `mode` is 'constant'.

    """
    if not isinstance(X, Process):
        raise TypeError('Data needs to be passed to skhyper.process.Process first')

    if X.n_dimension == 3:
        for _x in range(X.shape[0]):
            for _y in range(X.shape[1]):
                X.data[_x, _y, :] = _savgol_filter(X.data[_x, _y, :], window_length=window_length,
                                                   polyorder=polyorder, deriv=deriv, delta=delta,
                                                   axis=axis, mode=mode, cval=cval)

    elif X.n_dimension == 4:
        for _x in range(X.shape[0]):
            for _y in range(X.shape[1]):
                for _z in range(X.shape[2]):
                    X.data[_x, _y, _z, :] = _savgol_filter(X.data[_x, _y, _z, :], window_length=window_length,
                                                           polyorder=polyorder, deriv=deriv, delta=delta,
                                                           axis=axis, mode=mode, cval=cval)

    X._initialize()
    print('Hyperspectral data smoothed.')
