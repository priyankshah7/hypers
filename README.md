# scikit-hyper
[![Build Status](https://travis-ci.com/priyankshah7/scikit-hyper.svg?token=xX99xZvXU9jWErT5D1zh&branch=master)](https://travis-ci.com/priyankshah7/scikit-hyper)
[![Documentation Status](https://readthedocs.org/projects/scikit-hyper/badge/?version=latest)](http://scikit-hyper.readthedocs.io/en/latest/?badge=latest)
[![Python Version 3.5](https://img.shields.io/badge/Python-3.5-blue.svg)](https://www.python.org/downloads/)
[![Python Version 3.6](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/scikit-hyper.svg)](https://badge.fury.io/py/scikit-hyper)

Machine learning for hyperspectral data in Python

+ Simple tools for exploratory analysis of hyperspectral data
+ Built on numpy, scipy, matplotlib and scikit-learn
+ Simple to use, syntax similar to scikit-learn

<p align="center"><img src="/docs/images/hyperspectral_image.png" width="300"></p>

## Contents
1. [About](#about)
1. [Installation](#installation)
2. [Features](#features)
3. [Examples](#examples)
4. [Documentation](#documentation)
5. [License](#license)

## About
This package builds upon the popular scikit-learn to provide an interface for performing 
machine learning on hyperspectral data. Many of the commonly used techniques in the 
analysis of hyperspectral data (PCA, ICA, clustering and classification) have been 
implemented and more will be added in the future.

scikit-hyper also provides two features which aim to make exploratory analysis easier:

+ [Process object](http://scikit-hyper.readthedocs.io/en/latest/source/process/index.html) (`skhyper.process.Process`)
    
    This class forms the core of scikit-hyper. It provides useful information about the 
    hyperspectral data and makes machine learning on the data simple.
    
+ [Interactive hyperspectral viewer](http://scikit-hyper.readthedocs.io/en/latest/source/hypview/index.html)

    A lightweight pyqt gui that provides an interative interface to view the 
    hyperspectral data.
    
    <p align="center"><img src="/docs/source/hypview/hyperspectral_view.png" width="400"></p>
    
**Please note that this package is currently in pre-release. The first general release will 
be v0.1.0**

## Installation
To install using `pip`:
```
pip install scikit-hyper
```

The following packages are required:

+ numpy
+ scipy
+ scikit-learn
+ matplotlib
+ seaborn
+ PyQt5
+ pyqtgraph

## Features
Features implemented in scikit-hyper include:

+ [Classification](http://scikit-hyper.readthedocs.io/en/latest/source/classification/index.html) (e.g. SVM, Naive Bayes)
+ [Clustering](http://scikit-hyper.readthedocs.io/en/latest/source/cluster/index.html) (KMeans)
+ [Decomposition](http://scikit-hyper.readthedocs.io/en/latest/source/decomposition/index.html) (e.g. PCA, ICA, NMF)
+ [Hyperspectral viewer](http://scikit-hyper.readthedocs.io/en/latest/source/hypview/index.html)
+ [Tools](http://scikit-hyper.readthedocs.io/en/latest/source/tools/index.html) (smoothing, normalization)

	
## Examples

### Hyperspectral denoising
```python
import numpy as np
from skhyper.process import Process
from skhyper.decomposition import PCA

# Generating a random 4-d dataset and creating a Process instance
test_data = np.random.rand(200, 200, 10, 1024)
X = Process(test_data, scale=True)

# To denoise the dataset using PCA:
# First we fit the PCA model to the data, and then fit_transform()
# All the usual scikit-learn parameters are available
mdl = PCA()
mdl.fit_transform(X)

# The scree plot can be accessed by:
mdl.plot_statistics()

# Choosing the number of components to keep, we project back 
# into the original space:
Xd = mdl.inverse_transform(n_components=200)

# Xd is another instance of Process, which contains the new
# denoised hyperspectral data
```

### Hyperspectral clustering
```python
import numpy as np
from skhyper.process import Process
from skhyper.cluster import KMeans

# Generating a random 3-d dataset and creating a Process instance
test_data = np.random.rand(200, 200, 1024)
X = Process(test_data, scale=True)

# Again, all the usual scikit-learn parameters are available
mdl = KMeans(n_clusters=4)
mdl.fit(X)

# The outputs are:
# mdl.labels_  (a 2d/3d image with n_clusters number of labels)
# mdl.image_components_  (a list of n_clusters number of image arrays)
# mdl.spec_components_  (a list of n_clusters number of spectral arrays)
```

## Documentation
The docs are hosted [here](http://scikit-hyper.readthedocs.io/en/latest/?badge=latest).

The package API includes documentation from the [scikit-learn](https://github.com/scikit-learn/scikit-learn) 
modules where the particular module is wrapped around the scikit-learn version.

## License
scikit-hyper is licensed under the OSI approved BSD 3-Clause License.
