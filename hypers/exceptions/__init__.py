"""
Exposes public exceptions and warinings
"""


class DimensionError(ValueError):
    """
    Incorrect number of dimensions. Should be raised when an
    operation requires a certain number of dimensions and this
    requirement is not met.
    """
