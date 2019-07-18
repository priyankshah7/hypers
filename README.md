# hypers
[![Build Status](https://travis-ci.com/priyankshah7/hypers.svg?token=xX99xZvXU9jWErT5D1zh&branch=master)](https://travis-ci.com/priyankshah7/hypers)
[![Documentation Status](https://readthedocs.org/projects/hypers/badge/?version=latest)](http://hypers.readthedocs.io/en/latest/?badge=latest)
[![Python Version 3.5](https://img.shields.io/badge/Python-3.5-blue.svg)](https://www.python.org/downloads/)
[![Python Version 3.6](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/hypers.svg)](https://badge.fury.io/py/hypers)

hypers provides a data structure in python for hyperspectral data. The data structure includes:

+ Tools for processing and exploratory analysis of hyperspectral data
+ Interactive hyperspectral viewer (using PyQt) that can be accessed as a method from the object
+ Allows for unsupervised machine learning directly on the object

The data structure is built on top of the numpy `ndarray`, and this package simply adds additional functionality that 
allows for quick analysis of hyperspectral data. Importantly, this means that the object can still be used as a 
normal numpy array.

<p align="center"><img src="/docs/source/images/hyperspectral_image.png" width="300"></p>

**Please note that this package is currently in pre-release. It can still be used, however there is likely to be 
significant changes to the API. The first public release will be v0.1.0.**

## Contents
1. [Installation](#installation)
2. [Features](#features)
3. [Examples](#examples)
4. [Documentation](#documentation)
5. [License](#license)
   
## Installation
To install using `pip`:
```
pip install hypers
```

The following packages will also be installed:

+ numpy
+ matplotlib
+ scipy
+ scikit-learn
+ PyQt5
+ pyqtgraph

## Features
Features implemented in ``hypers`` include:

+ Clustering
+ Decomposition (e.g. PCA, ICA, NMF)
+ Hyperspectral viewer
+ Vertex component analysis
+ Gaussian mixture models

A full list of features can be found [here](http://hypers.readthedocs.io/en/latest/).
	
## Examples

### Hyperspectral dimensionality reduction and clustering
Below is a quick example of using some of the features of the package on a randomized hyperspectral array. 
For an example using the IndianPines dataset, see the Jupyter notebook in the [examples](/examples/indian_pines.ipynb) directory.

```python
import numpy as np
import hypers as hp

# Generating a random 4-d dataset and creating a Dataset instance
# The test dataset here has spatial dimensions (x=200, y=200, z=10) and spectral dimension (s=1024)
test_data = np.random.rand(200, 200, 10, 1024)
X = hp.array(test_data)

# Using Principal Components Analysis to reduce to first 5 components
# The variables ims, spcs are arrays of the first 5 principal components for the images, spectra respectively
ims, spcs = X.decompose.pca.calculate(n_components=5)

# Clustering using K-means (with and without applying PCA first)
# The cluster method will return the labeled image array and the spectrum for each cluster
lbls_nodecompose, spcs_nodecompose = X.cluster.kmeans.calculate(
    n_clusters=3,
    decomposed=False
)

# Clustering on only the first 5 principal components
lbls_decomposed, spcs_decomposed = X.cluster.kmeans.calculate(
    n_clusters=3,
    decomposed=True,
    pca_comps=5
)
```

### Interactive viewer
The interactive viewer can be particularly helpful for exploring a completely new dataset for the first time to get 
a feel for the type of data you are working with. An example from a coherent anti-Stokes Raman (CARS) dataset is 
shown below:
 
 <p align="center"><img src="/docs/source/images/hyperspectral_view.png" width="400"></p>

## Documentation
The docs are hosted [here](http://hypers.readthedocs.io/en/latest/?badge=latest).

## License
hypers is licensed under the OSI approved BSD 3-Clause License.
