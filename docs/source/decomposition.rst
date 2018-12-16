========================
Dimensionality Reduction
========================

Dimensionality reduction with ``hypers`` is done directly on the Process object model.
At the moment, the following classes from ``scikit-learn`` are supported:

- PCA
- IncrementalPCA
- TruncatedSVD
- FastICA
- DictionaryLearning
- MiniBatchDictionaryLearning
- FactorAnalysis
- NMF
- LatentDirichletAllocation

.. code-block:: python

    import numpy as np
    import hypers as hp
    from sklearn.decomposition import PCA

    tst_data = np.random.rand(50, 50, 1000)
    X = hp.Dataset(tst_data)

    # Retrieving images and spectra of the first 10 principal components of the dataset
    ims, spcs = X.decompose(
        mdl=PCA(n_components=10)
    )
