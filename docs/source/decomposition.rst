========================
Dimensionality Reduction
========================

Dimensionality reduction with ``scikit-hyper`` is done directly on the Process object model. 
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
    from sklearn.decomposition import PCA

    from skhyper.process import Process 

    tst_data = np.random.rand(50, 50, 1000)
    X = Process(tst_data, scale=True)

    # Retrieving images and spectra of the first 10 principal components of the dataset
    ims, spcs = X.decompose(
        mdl=PCA(n_components=10)
    )
