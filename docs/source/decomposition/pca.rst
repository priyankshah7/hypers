============================
Principal component analysis
============================

:class:`~skhyper.decomposition.PCA` is used to decompose a multivariate
dataset in a set of successive orthogonal components that explain a
maximum amount of the variance.

The optional parameter ``whiten=True`` makes it possible to project the data
onto the singular space while scaling each component to unit variance.
This is often useful if the models down-stream make strong assumptions
on the isotropy of the signal: this is for example the case for Support
Vector Machines with the RBF kernel and the K-Means clustering algorithm.

**Example**

.. code-block:: python

    import numpy as np
    from skhyper.process import Process
    from skhyper.decomposition import PCA

    random_data = np.random.rand(250, 250, 1024)
    X = Process(random_data, scale=True)

    mdl = PCA()
    mdl.fit_transform(X)

    # To access model statistics including the scree plot:
    mdl.plot_statistics()

    # Xd is another Process instance that contains the denoised data
    Xd = inverse_transform(n_components=200)
