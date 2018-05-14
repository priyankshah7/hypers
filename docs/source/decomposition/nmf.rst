=================================
Non-negative matrix factorization
=================================

:class:`~skhyper.decomposition.NMF` is an alternative approach to decomposition that
assumes that the data and the components are non-negative. It can be used in place of
:class:`~skhyper.decomposition.PCA` or its variants, in the cases where the spectral
data do no contain negative values.

Whilst :class:`~skhyper.decomposition.NMF` can be used in place of :class:`~skhyper.decomposition.PCA`,
i.e. for denoising hyperspectral datasets, it is often used to directly retrieve
the actual components present in the dataset. For example, assuming that there are
three classes present in the dataset, the first 3 components obtained by :class:`~skhyper.decomposition.NMF`
often correlate well with the true classes [ref]. The example below shows how to
achieve this using this package.

**Example**

.. code-block:: python

    import numpy as np
    from skhyper.process import Process
    from skhyper.decomposition import NMF

    random_data = np.random.rand(250, 250, 1024)
    X = Process(random_data, scale=True)

    # Retrieve the first 3 components from the decomposition
    mdl = NMF(n_components=3)
    mdl.fit_transform(X)

    # The spectra and image of the 3 components are then obtained from:
    mdl.image_components[]
    mdl.spec_components[]
