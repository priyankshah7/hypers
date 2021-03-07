# hypers
[![Build Status](https://travis-ci.com/priyankshah7/hypers.svg?token=xX99xZvXU9jWErT5D1zh&branch=master)](https://travis-ci.com/priyankshah7/hypers)
[![Documentation Status](https://readthedocs.org/projects/hypers/badge/?version=latest)](http://hypers.readthedocs.io/en/latest/?badge=latest)
[![Python Version 3.5+](https://img.shields.io/badge/Python-3.5%2B-blue)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/hypers.svg)](https://badge.fury.io/py/hypers)

hypers provides a data structure in python for hyperspectral data. The data structure includes:

+ Tools for processing and exploratory analysis of hyperspectral data
+ Interactive hyperspectral viewer (using PyQt) that can be accessed as a method from the object
+ Allows for unsupervised machine learning directly on the object

The data structure is built on top of the numpy `ndarray`, and this package simply adds additional functionality that 
allows for quick analysis of hyperspectral data. Importantly, this means that the object can still be used as a 
normal numpy array.

[//]: <> (<p align="center"><img src="/docs/source/images/hyperspectral_image.png" width="300"></p>)

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
+ scipy
+ PyQt5
+ pyqtgraph

## Features
Features implemented in ``hypers`` include:

+ Hyperspectral viewer
+ Vertex component analysis
+ Abundance mapping

A full list of features can be found [here](http://hypers.readthedocs.io/en/latest/).
	
## Examples

### Interactive viewer
The interactive viewer can be particularly helpful for exploring a completely new dataset for the first time to get 
a feel for the type of data you are working with. An example from a coherent anti-Stokes Raman (CARS) dataset is 
shown below:
 
 <p align="center"><img src="/docs/source/images/hyperspectral_view.png" width="400"></p>

## Documentation
The docs are hosted [here](http://hypers.readthedocs.io/en/latest/?badge=latest).

## License
hypers is licensed under the OSI approved BSD 3-Clause License.

## References
1. VCA algorithm  
J. M. P. Nascimento and J. M. B. Dias, "Vertex component analysis: a fast algorithm to unmix hyperspectral data," 
in IEEE Transactions on Geoscience and Remote Sensing, 2005  
Adapted from [repo](https://github.com/Laadr/VCA).