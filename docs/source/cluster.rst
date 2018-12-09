==========
Clustering
==========

Clustering data with ``scikit-hyper`` is done directly on the Process object model. At the 
moment, the following clustering classes from ``scikit-learn`` are supported:

- KMeans
- AgglomerativeClustering
- SpectralClustering

Clustering can be performed on both the data stored in the Process object itself, or on a 
set of principal components of the dataset (as demonstrated below).

.. code-block:: python
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA 

    from skhyper.process import Process 

    tst_data = np.random.rand(50, 50, 1000)
    X = Process(tst_data, scale=True)

    # Clustering on the stored data first
    lbls_nodecompose, spcs_nodecompose = X.cluster(
        mdl=KMeans(n_clusters=4),
        decomposed=False
    )

    # Clustering on the first 5 principal components on the dataset
    lbls_decomposed, spcs_decomposed = X.cluster(
        mdl=KMeans(n_clusters=5),
        decomposed=True,
        pca_comps=5
    )
