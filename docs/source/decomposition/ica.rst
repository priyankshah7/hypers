==============================
Independent component analysis
==============================

Independent component analysis separate a multivariate signal into additive
subcomponents that are maximally independent. It is implemented here in
:class:`~skhyper.decomposition.FastICA` using `scikit-learn`'s version.
As with :class:`~skhyper.decomposition.NMF`, ICA is not usually used
for reducing dimensionality but for separating superimposed signals.

Since the ICA model does not include a noise term, for the model to be correct,
whitening must be applied. This can be done internally using the whiten
argument or manually using one of the PCA variants.

**Example**

.. code-block:: python

    import numpy as np
    from skhyper.process import Process
    from skhyper.decomposition import FastICA

    random_data = np.random.rand(250, 250, 1024)
    X = Process(random_data, scale=True)

    # Retrieve the first 3 components from the decomposition
    mdl = FastICA(n_components=3)
    mdl.fit_transform(X)

    # The spectra and image of the 3 components are then obtained from:
    mdl.image_components[]
    mdl.spec_components[]
