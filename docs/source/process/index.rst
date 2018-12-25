======================
Dataset data structure
======================

Processing data
===============
The hyperspectral data is stored and processed using :class:`~hypers.Dataset`.

.. note::

    Note that the Dataset class will only accept a ``numpy`` array of dimensions 3 or 4. The array should be
    formatted as:

    `(x, y, spectrum)` or `(x, y, z, spectrum)`

Below is an example of instantiating a ``Dataset`` object with a 4d random numpy array.

.. code-block:: python

    import numpy as np
    import hypers as hp

    test_data = np.random.rand(40, 40, 4, 512)
    X = hp.Dataset(test_data)

Dataset properties
==================
The ``Dataset`` object has several useful attributes and methods for immediate analysis:

.. code-block:: python

    # Data properties:
    X.shape                            # Shape of the hyperspectral array
    X.ndim                             # Number of dimensions (3 or 4)
    X.n_features                       # Number of spectral points (features)
    X.n_samples                        # Total number of pixels (samples)

    # To access the mean image/spectrum of the dataset:
    X.mean_spectrum
    X.mean_image

    # To access the image/spectrum in a specific pixel/spectral range:
    X.spectrum[10:20, 10:20, :, :]     # Returns spectrum within chosen pixel range
    X.image[..., 100:200]              # Returns image averaged between spectral bands

    # To access the scree plot (as an array) that explains the variance contribution:
    X.scree()

    # To view and interact with the data:
    X.view()                           # Opens a hyperspectral viewer

The ``Dataset`` object also supports arithmetic operations in the following manner:

.. code-block:: python

    import numpy as np
    import hypers as hp

    test_data = np.random.rand(50, 50, 512)
    spectral_array = np.random.rand(512)

    X = hp.Dataset(test_data)

    # For arithmetic operations with a constant (int or float)
    # This will be performed element-wise (on every single spectral band at every single pixel)
    X *= 2
    X /= 2
    X += 2
    X -= 2

    # For arithmetic operations with a spectrum (spectral_array must have the same size as the spectra in Dataset)
    # This will be performed on every single spectrum at every pixel
    X *= spectral_array
    X /= spectral_array
    X += spectral_array
    X -= spectral_array

To view the full list of methods and attributes that the Process class contains, see 
:class:`~hypers.Dataset`.
