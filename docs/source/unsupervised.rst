=====================
Unsupervised learning
=====================

The ``hparray`` object has built in methods that allows you to perform several unsupervised learning
techniques on the stored data. The techniques are split into the following categories:

- Dimensionality reduction
- Cluster analysis
- Mixture models
- Abundance mapping

These are all available as methods on the ``hparray`` object.

.. code-block:: python

    import numpy as np
    import hypers as hp

    test_data = np.random.rand(10, 10, 1000)
    X = hp.array(test_data)

    # To access PCA from the dimensionality reduction techniques
    ims, spcs = X.decompose.pca.calculate(n_components=10)

    # To access k-means from the clustering techniques
    lbls, spcs = X.cluster.kmeans.calculate(n_clusters=4)

    # To access Gaussian mixture models
    lbls, spcs = X.mixture.gaussian_mixture.calculate(n_components=10)

    # To access unconstrained least-squares for abundance mapping
    spectra = np.random.rand(1000, 2)
    amap = X.abundance.ucls.calculate(spectra)


Dimensionality reduction
========================
Dimensionality reduction refers to techniques that obtain a set of principal components that explain
the data in some pre-defined way. The class of techniques are particularly useful for hyperspectral data as
the data can often be described using only a very small subset of the generated principal components and thus
serve as a way of "reducing" dimensionality.

The following techniques are available:

- Principal components analysis
- Independant components analysis
- Non-negative matrix factorization
- Vertex component analysis

Principal component analysis
----------------------------
This is implemented using scikit-learn's ``PCA`` and thus requires scikit-learn to be installed.

.. autoclass:: hypers.learning.decomposition.pca
    :members:

Independent component analysis
------------------------------
This is implemented using scikit-learn's ``FastICA`` and thus requires scikit-learn to be installed.

.. autoclass:: hypers.learning.decomposition.ica
    :members:

Non-negative matrix factorization
---------------------------------
This is implemented using scikit-learn's ``NMF`` and thus requires scikit-learn to be installed.

.. autoclass:: hypers.learning.decomposition.nmf
    :members:

Vertex component analysis
-------------------------
This is implemented with [1]_.

.. autoclass:: hypers.learning.decomposition.vca
    :members:

------------

Cluster analysis
================
Clustering refers to grouping objects into a set number of clusters whilst ensuring that the objects in
each cluster are as similar as possible to the other objects in the cluster. The notion of "similarity" is where
the different clustering techniques differ.

The following techniques are available:

- K-means

K-means
-------
This is implemented using scikit-learn's ``KMeans`` and thus requires scikit-learn to be installed.

.. autoclass:: hypers.learning.cluster.kmeans
    :members:

------------

Mixture models
==============
Mixture models are probabilistic models that assume that every point in a dataset is generated from a
mixture of finite number of probability distributions with unknown parameters.

The following techniques are available:

- Gaussian mixture model

Gaussian mixture model
----------------------
This is implemented using scikit-learn's ``GaussianMixture`` and thus requires scikit-learn to be installed.

.. autoclass:: hypers.learning.mixture.gaussian_mixture
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

