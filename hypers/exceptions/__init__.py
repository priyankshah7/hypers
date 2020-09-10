"""
Exposes public exceptions and warnings
"""
import warnings

__all__ = ['DimensionError', 'warning_insufficient_samples']


class DimensionError(ValueError):
    """
    Incorrect number of dimensions. Should be raised when an
    operation requires a certain number of dimensions and this
    requirement is not met.
    """


def warning_insufficient_samples_format(message, category, filename, lineno, file=None, line=None):
    return '%s: %s' % ('InsufficientSamplesWarning', message)


def warning_insufficient_samples(nsamples: int, nspectral: int):
    """
    Warning: total no. of samples is less than total no. of spectral bands

    Parameters
    ----------
    nsamples: int
        Total number of samples (spatial pixels)
    nspectral: int
        Total number of spectral bands
    """
    warnings.formatwarning = warning_insufficient_samples_format
    warnings.warn(f'Total number of pixels ({nsamples}) is less than total number of spectral bands ({nspectral})')


def warning_insufficient_dimensions_format(message, category, filename, lineno, file=None, line=None):
    return '%s: %s' % ('InsufficientDimensionsWarning', message)


def warning_insufficient_dimensions(nsamples: int, nspectral: int):
    """
    Warning: insufficient number of dimensions. Must be 2 or greater.

    The hp.hparray is suitable for arrays with a dimension of 2 or greater with
    a format of ([spatial_dims], spectral_dim). If a 1D array is provided,

    Parameters
    ----------
    nsamples: int
        Total number of samples (spatial pixels)
    nspectral: int
        Total number of spectral bands
    """
    warnings.formatwarning = warning_insufficient_samples_format
    warnings.warn(f'Total number of pixels ({nsamples}) is less than total number of spectral bands ({nspectral})')
