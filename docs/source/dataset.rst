=========================
hparray: An  introduction
=========================

Motivation
==========
The motivation behind the creation of this package was performing common tasks on a numpy ``ndarray`` for
hyperspectral data which could be better served by extending the ``ndarray`` type with added functionality for
hyperspectral data. This package provides just that, a :class:`~hypers.core.array.hparray` type that subclasses
``ndarray`` and adds further functionality. An advantage over other packages is that it the
:class:`~hypers.core.array.hparray` object can still be used as a normal numpy array for other tasks.

Processing data
===============
The hyperspectral data is stored and processed using :class:`~hypers.core.array.hparray`.

.. note::

    Note that the array should be formatted in the following order:

    `(spatial, spectral)`

    i.e. the spatial dimensions should proceed the spectral dimension/channels. As an example, if our
    hyperspectral dataset has dimensions of `x=10`, `y=10`, `z=10` and `channels=100` then the array should be
    formatted as:

    `(10, 10, 10, 100)`


Below is an example of instantiating a :class:`~hypers.core.array.hparray` object with a 4d random numpy array.

.. code-block:: python

    import numpy as np
    import hypers as hp

    test_data = np.random.rand(40, 40, 4, 512)
    X = hp.array(test_data)

Properties
==========
The :class:`~hypers.core.array.hparray` object has several useful attributes and methods for immediate analysis:

.. note::

    Note that as :class:`~hypers.core.array.hparray` subclasses numpy's ``ndarray``, all the usual methods
    and attributes in a numpy array can also be used here.

.. code-block:: python

    # Data properties:
    X.shape                            # Shape of the hyperspectral array
    X.ndim                             # Number of dimensions
    X.nfeatures                        # Size of the spectral dimension/channels
    X.nsamples                         # Total number of pixels (samples)
    X.nspatial                         # Shape of the spatial dimensions

    # To access the mean image/spectrum of the dataset:
    X.mean_spectrum
    X.mean_image

    # To access the image/spectrum in a specific pixel/spectral range:
    X.spectrum[10:20, 10:20, :, :]     # Returns spectrum within chosen pixel range
    X.image[..., 100:200]              # Returns image averaged between spectral bands

    # To access the scree plot that explains the variance contribution of the principal components:
    X.decompose.pca.plot_scree()

    # To view and interact with the data:
    X.plot(backend='pyqt')                           # Opens a hyperspectral viewer


To view the full list of methods and attributes that the Process class contains, see
:class:`~hypers.core.array.hparray`.
