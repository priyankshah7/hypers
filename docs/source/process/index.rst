===============
Processing data
===============

Processing data is done using :class:`~skhyper.process.Process`. Note that this requires the
hyperspectral array to have been formatted into a numpy array.

.. code-block:: python

    import numpy as np
    from skhyper.process import Process

    test_data = np.random.rand(40, 40, 4, 512)
    X = Process(test_data, scale=True)

The above example passes a 4D (random) hyperspectral numpy array into the `Process` instance `X`.
The object `X` has several useful attributes for immediate analysis:


.. code-block:: python

    # Data properties:
    X.shape                            # Shape of the hyperspectral array
    X.n_dimension                      # Number of dimensions (3 or 4)
    X.n_features                       # Number of spectral points (features)
    X.n_samples                        # Total number of pixels (samples)

    # To access the mean image/spectrum of the dataset:
    X.mean_spectrum
    X.mean_image

    # To access the image/spectrum in a specific pixel/spectral range:
    X.spectrum[10:20, 10:20, :, :]     # Returns spectrum within chosen pixel range
    X.image[..., 100:200]              # Returns image averaged between spectral bands

    # To view and interact with the data:
    X.view()                           # Opens a hyperspectral viewer

