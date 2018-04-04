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
```python
pip install scikit-hyper
```

## Tools

+ Current techniques implemented.
+ Future proposed techniques to implement.

## Example
```python
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
```

## Hyperspectral Viewer
```python
import numpy as np
from skhyper import hsiPlot

random_data = np.random.rand(100, 100, 10, 1024)

# Opening the hyperspectral viewer
hsiPlot(random_data)
```
