.. scikit-hyper documentation master file, created by
   sphinx-quickstart on Tue May  8 13:44:23 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

scikit-hyper
============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Machine learning for hyperspectral data in Python

-  Simple tools for exploratory analysis of hyperspectral data
-  Built on numpy, scipy, matplotlib and scikit-learn
-  Simple to use, syntax similar to scikit-learn

Installation
============

To install using ``pip``:

.. code:: python

   pip install scikit-hyper

Features
========

- Cluster
   - ``KMeans``: K-means clustering - `completed`
   - ``AgglomerativeClustering``: Hierarchical clustering - `todo`
   - ``DBSCAN``: Density-Based Spatial Clustering of Applications with Noise - `todo`

- Decomposition
   - ``PCA``: Principal components analysis - `completed`
   - ``KernelPCA``: Kernel principal components analysis - `todo`
   - ``ICA``: Independent component analysis - `completed`
   - ``NMF``: Non-negative matrix factorization - `completed`

- Tools
   - ``savgol_filter``: Savitzky-Golay filter (spectral smoothing) - `completed`
   - ``normalize``: Spectral normalization - `completed`


Upcoming features
-----------------

- Classification (supervised)
   - ``SVC``: Support vector machines
   - ``KNeighborsClassifier``: K-nearest neighbors
   - ``GaussianNB``: Gaussian Naive Bayes
   - ``MLPClassifier``: Multi-layer perceptron neural network


Examples
========

.. code:: python

   import numpy as np
   from skhyper.cluster import KMeans

   # Generating a random 3-dimensional hyperspectral dataset
   test_data = np.random.rand(200, 200, 1024)

   # Retrieving 3 clusters using KMeans from the hyperspectral dataset
   mdl = KMeans(n_clusters=3)
   mdl.fit(test_data)

   data_clusters = mdl.data_clusters
   labels = mdl.labels

   # Plotting the retrieved clusters (plots the associated image and spectrum of each cluster)
   mdl.plot()


Hyperspectral Viewer
====================

.. code:: python

   import numpy as np
   from skhyper import hsiPlot

   random_data = np.random.rand(100, 100, 10, 1024)

   # Opening the hyperspectral viewer
   hsiPlot(random_data)

.. figure:: /images/hyp_view.png
   :alt: HyperspectralViewer




API
===
* :ref:`modindex`

