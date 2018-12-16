# hypers
[![Build Status](https://travis-ci.com/priyankshah7/hypers.svg?token=xX99xZvXU9jWErT5D1zh&branch=master)](https://travis-ci.com/priyankshah7/hypers)
[![Documentation Status](https://readthedocs.org/projects/hypers/badge/?version=latest)](http://hypers.readthedocs.io/en/latest/?badge=latest)
[![Python Version 3.5](https://img.shields.io/badge/Python-3.5-blue.svg)](https://www.python.org/downloads/)
[![Python Version 3.6](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/hypers.svg)](https://badge.fury.io/py/hypers)

Provides an object model for hyperspectral data.

+ Simple tools for exploratory analysis of hyperspectral data
+ Interactive hyperspectral viewer built into the object
+ Allows for unsupervised machine learning directly on the object (using scikit-learn)
+ More features coming soon...

<p align="center"><img src="/docs/images/hyperspectral_image.png" width="300"></p>

## Contents
1. [About](#about)
1. [Installation](#installation)
2. [Features](#features)
3. [Examples](#examples)
4. [Documentation](#documentation)
5. [License](#license)

## About
This package provides an object model for hyperspectral data (e.g. similar to pandas for tabulated data). Many of the 
commonly used tools are built into the object, including a lightweight interactive gui for visualizing the data. 
Importantly, the object also interfaces with `scikit-learn` to allow the cluser and decomposition classes (e.g. PCA, 
ICA, K-means) to be used directly with the object.

+ [Dataset object](http://hypers.readthedocs.io/en/latest/source/Dataset/index.html) (`hypers.Dataset`)
    
    This class forms the core of hypers. It provides useful information about the 
    hyperspectral data and makes machine learning on the data simple.
    
+ [Interactive hyperspectral viewer](http://hypers.readthedocs.io/en/latest/source/hypview/index.html)

    A lightweight pyqt gui that provides an interative interface to view the 
    hyperspectral data.
    
    <p align="center"><img src="/docs/source/hypview/hyperspectral_view.png" width="400"></p>
    
**Please note that this package is currently in pre-release. The first general release will 
be v0.1.0**

#### Hyperspectral data
Whilst this package is designed to work with any type of hyperspectral data, of the form: 

**X[x, y, spectrum]** or
**X[x, y, z, spectrum]**

some of the features are particularly useful for vibrational-scattering related hyperspectral data (e.g. Raman micro-spectroscopy), e.g. the spectral component of the hyperspectral viewer (see figure above).


## Installation
To install using `pip`:
```
pip install hypers
```

The following packages are required:

+ numpy
+ scipy
+ scikit-learn
+ PyQt5
+ pyqtgraph

## Features
Features implemented in ``hypers`` include:

+ [Clustering](http://hypers.readthedocs.io/en/latest/source/cluster/index.html) (e.g. KMeans, Spectral clustering, Hierarchical clustering)
+ [Decomposition](http://hypers.readthedocs.io/en/latest/source/decomposition/index.html) (e.g. PCA, ICA, NMF)
+ [Hyperspectral viewer](http://hypers.readthedocs.io/en/latest/source/hypview/index.html)

	
## Examples

### Hyperspectral dimensionality reduction and clustering
Below is a quick example of using some of the features of the package on a randomized hyperspectral array. For an example using the IndianPines dataset, see the Jupyter notebook in the examples/ directory.

```python
import numpy as np
import hypers as hp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Generating a random 4-d dataset and creating a Process instance
# The test dataset here has spatial dimensions (x=200, y=200, z=10) and spectral dimension (s=1024)
test_data = np.random.rand(200, 200, 10, 1024)
X = hp.Dataset(test_data)

# Using Principal Components Analysis to reduce to first 5 components
# The variables ims, spcs are arrays of the first 5 principal components for the images, spectra respectively
ims, spcs = X.decompose(
    mdl=PCA(n_components=5)
)

# Clustering using K-means (with and without applying PCA first)
# The cluster method will return the labeled image array and the spectrum for each cluster
lbls_nodecompose, spcs_nodecompose = X.cluster(
    mdl=KMeans(n_clusters=3),
    decomposed=False
)

# Clustering on only the first 5 principal components
lbls_decomposed, spcs_decomposed = X.cluster(
    mdl=KMeans(n_clusters=3),
    decomposed=True,
    pca_comps=5
)
```

## Documentation
The docs are hosted [here](http://hypers.readthedocs.io/en/latest/?badge=latest).

## License
hypers is licensed under the OSI approved BSD 3-Clause License.
