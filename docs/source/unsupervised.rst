=====================
Unsupervised learning
=====================

The ``hparray`` object has built in methods that allows you to perform several unsupervised learning
techniques on the stored data. The techniques are split into the following categories:

- Spectral unmixing
- Abundance mapping

These are all available as methods on the ``hparray`` object.

.. code-block:: python

    import numpy as np
    import hypers as hp

    test_data = np.random.rand(10, 10, 1000)
    X = hp.array(test_data)

    # To access vertex component analysis
    ims, spcs = X.unmix.vca.calculate(n_components=10)

    # To access unconstrained least-squares for abundance mapping
    spectra = np.random.rand(1000, 2)
    amap = X.abundance.ucls.calculate(spectra)

Spectral unmixing
=================
Spectral unmixing is the process of decomposing the spectral signature of a
mixed pixel into a set of endmembers and their corresponding abundances.

The following techniques are available:

- Vertex component analysis

Vertex component analysis
-------------------------
This is implemented with [1]_.

.. autoclass:: hypers.learning.decomposition.vca
    :members:

------------

Abundance mapping
=================
Abundance maps are used to determine how much of a given spectrum is present at each pixel in a hyperspectral
image. They can be useful for determining percentages after the spectra have been retrieved from some clustering
or unmixing technique or if the spectra are already at hand.

The following techniques are available:

- Unconstrained least-squares
- Non-negative constrained least-squares
- Fully-constrained least-squares

Unconstrained least-squares
---------------------------
This is implemented with [2]_.

.. autoclass:: hypers.learning.abundance.ucls
    :members:

Non-negative constrained least-squares
--------------------------------------
This is implemented with [2]_.

.. autoclass:: hypers.learning.abundance.nnls
    :members:

Fully-constrained least-squares
-------------------------------
This is implemented with [2]_.

.. autoclass:: hypers.learning.abundance.fcls
    :members:

------------

**References**

.. [1] VCA algorithm.
    J. M. P. Nascimento and J. M. B. Dias, "Vertex component analysis: a fast algorithm to unmix hyperspectral data,"
    in IEEE Transactions on Geoscience and Remote Sensing, 2005.
    Adapted from repo_.

.. [2] Abundance mapping.
    Adapted from PySptools_.

.. _PySptools: https://github.com/ctherien/pysptools
.. _repo: https://github.com/Laadr/VCA

