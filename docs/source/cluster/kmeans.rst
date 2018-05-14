=======
K-means
=======

The :class:`~skhyper.cluster.KMeans` algorithm clusters data by trying to separate
samples into `n` groups of equal variance, minimizing a criterion known as the inertia
or within-cluster sum-of-squares. The algorithm requres the number of clusters to be
specified. It scales well to large number of samples.

The :class:`~skhyper.cluster.KMeans` algorithm can take all the same parameters
as in the `scikit-learn` version.

**Example**

.. code-block:: python

    import numpy as np
    from skhyper.process import Process
    from skhyper.cluster import KMeans

    random_data = np.random.rand(250, 250, 1024)
    X = Process(random_data, scale=True)

    mdl = KMeans(n_clusters=3)
    mdl.fit(X)

    # To access the retrieved clusters from the model:
    mdl.labels_                 # A 2d/3d image with pixels assigned to a cluster
    mdl.image_components_       # List of image array of each cluster
    mdl.spec_components_        # List of spectral array of each cluster
