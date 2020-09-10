import numpy as np

__all__ = ['map_units']


def map_units(current: list, future: list, tomap: list, deg: int = 1) -> np.ndarray:
    """
    Map units using a polynomial fit.

    Parameters
    ----------
    current: array_like
        List of values that are to be mapped
    future: array_like
        List of values that are to be mapped to
    tomap: array_like
        Array to perform mapping on
    deg: int
        Degree of the fitting polynomial

    Returns
    -------
    np.ndarray
        Mapped array
    """
    p = np.polyfit(current, future, deg=deg)
    mapped = np.polyval(p, tomap)
    return mapped
