# scikit-hyper
[![Build Status](https://travis-ci.com/priyankshah7/scikit-hyper.svg?token=xX99xZvXU9jWErT5D1zh&branch=master)](https://travis-ci.com/priyankshah7/scikit-hyper)
[![Python Version 3.5](https://img.shields.io/badge/Python-3.5-blue.svg)](https://www.python.org/downloads/)
[![Python Version 3.6](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/downloads/)

Machine learning for hyperspectral data in Python

+ Simple tools for exploratory analysis of hyperspectral data
+ Built on numpy, scipy, matplotlib and scikit-learn
+ Simple to use, syntax similar to scikit-learn

## Installation
To install using `pip`:
```
pip install scikit-hyper
```

## Features

+ Cluster
	+ KMeans clustering - completed
	+ Hierarchical clustering - todo
	+ DBSCAN - todo

+ Decomposition
	+ PCA - completed
	+ Kernel PCA - todo
	+ ICA - completed
	+ NMF - todo

+ Tools
	+ Spectral smoothing - completed
	+ Spectral normalization - completed

	
## Examples

### Hyperspectral denoising
```python
import numpy as np
from skhyper.process import Process
from skhyper.decomposition import PCA

# Generating a random 4-d dataset and creating a Process instance
test_data = np.random.rand(200, 200, 10, 1024)
X = Process(test_data, scale=True)

# The object X contains several useful features to explore the data
# e.g.
# X.mean_spectrum and X.mean_image (mean image/spectrum of the entire dataset)
# X.spectrum[:, :, :, :] and X.image[:, :, :, :] (image/spectrum in chosen region)
# X.view()  (opens a hyperspectral viewer with X loaded)
# X.flat (a 'flattened' 2-d version of the data)
# for all features, see the documentation

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
