# scikit-hyper
[![Build Status](https://travis-ci.com/priyankshah7/scikit-hyper.svg?token=xX99xZvXU9jWErT5D1zh&branch=master)](https://travis-ci.com/priyankshah7/scikit-hyper)
[![Python Version 3.5](https://img.shields.io/badge/Python-3.5-blue.svg)](https://www.python.org/downloads/)
[![Python Version 3.6](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/downloads/)

*Author: Priyank Shah* <br />
*Author email: priyank.shah@kcl.ac.uk* <br />
*Institution: King's College London* <br />
*Description: Hyperspectral data analysis package*

### Description
A python package that provides tools for easy exploration and analysis of hyperspectral data.

### Usage
```python
import numpy as np
import hyperanalysis as hyp

randomData = np.random.rand(50, 50, 10, 1024)

# Opening the hyperspectral viewer
hyp.hsiPlot(randomData)
```